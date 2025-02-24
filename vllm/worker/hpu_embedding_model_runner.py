import torch
import habana_frameworks.torch as htorch
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.worker.hpu_model_runner import HPUModelRunnerBase, ModelInputForHPU
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.pooling_params import PoolingParams

@dataclasses.dataclass(frozen=True)
class ModelInputForHPUWithPoolingMetadata(ModelInputForHPU):
    """
    Used by the HPUEmbeddingModelRunner.
    """
    pooling_metadata: Optional["PoolingMetadata"] = None

class HPUEmbeddingModelRunner(
        HPUModelRunnerBase[ModelInputForHPUWithPoolingMetadata]):
    _model_input_cls: Type[ModelInputForHPUWithPoolingMetadata] = (
        ModelInputForHPUWithPoolingMetadata)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_embedding = True

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForHPUWithPoolingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        warmup_mode=False
    ) -> Optional[Union[List[PoolerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "HPUEmbeddingModelRunner does not support multi-step execution.")

        input_tokens = model_input.input_tokens
        assert input_tokens is not None
        input_positions = model_input.input_positions
        assert input_positions is not None
        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None
        is_prompt = attn_metadata.is_prompt
        assert is_prompt is True

        seq_len = self._seq_len(attn_metadata)
        batch_size = input_tokens.size(0)

        real_batch_size = model_input.real_batch_size
        batch_size_padded = model_input.batch_size_padded

        use_graphs = self._use_graphs(batch_size, seq_len, is_prompt)
        self._check_config(batch_size, seq_len, is_prompt, warmup_mode)

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [ None for _ in range(num_layers)]

        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": self.trim_attn_metadata(model_input.attn_metadata),
            "intermediate_tensors": intermediate_tensors,
            "selected_token_indices": None
        }

        if htorch.utils.internal.is_lazy():
            execute_model_kwargs.update({"bypass_hpu_graphs": not use_graphs})

        if self.is_driver_worker:
            model_event_name = ("model_"
                                f"{'prompt' if is_prompt else 'decode'}_"
                                f"bs{batch_size}_"
                                f"seq{seq_len}_"
                                f"graphs{'T' if use_graphs else 'F'}")
        else:
            model_event_name = 'model_executable'
        
        htorch.core.mark_step()

        if self.is_driver_worker and self.profiler.enabled:
            # Stop recording 'execute_model' event
            self.profiler.end()
            event_end = self.profiler.get_timestamp_us()
            counters = self.profiler_counter_helper.get_counter_dict(
                cache_config=self.cache_config,
                duration=event_end - self.event_start,
                seq_len=seq_len,
                batch_size_padded=batch_size_padded,
                real_batch_size=real_batch_size,
                is_prompt=is_prompt)
            self.profiler.record_counter(self.event_start, counters)

        with self.profiler.record_event('internal', model_event_name):
            hidden_or_intermediate_states = self.model.forward(
                **execute_model_kwargs)

        htorch.core.mark_step()  

        outputs = [] 

        if self.is_driver_worker:
            outputs = [self.model.pooler(hidden_states=hidden_or_intermediate_states,
                              pooling_metadata=model_input.pooling_metadata)]    

        return outputs

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForHPUWithPoolingMetadata:
        return ModelInputForHPUWithPoolingMetadata.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForHPUWithPoolingMetadata:
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            assert seq_group_metadata_list is not None
            if self.profiler.enabled:
                self.profiler_counter_helper.capture_seq_group_metadata_stats(
                    seq_group_metadata_list=seq_group_metadata_list)

            # model_input, sampling_metadata = self.prepare_input_tensors(
            #     seq_group_metadata_list, finished_requests_ids)

            real_batch_size = len(seq_group_metadata_list)
            is_prompt = seq_group_metadata_list[0].is_prompt
            batch_size_padded = self.bucketing_ctx.get_padded_batch_size(
            real_batch_size, is_prompt)

            prefill_reqs = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_reqs.append(seq_group_meta)

            (
                input_tokens,
                input_positions,
                prefill_attn_metadata,
                seq_lens,
                query_lens,
                lora_index_mapping,
                lora_prompt_mapping,
                lora_requests,
                multi_modal_kwargs,
                slot_mapping,
                lora_ids,
            ) = self._prepare_prompt(prefill_reqs)

            model_input = self._model_input_cls(
                input_tokens=input_tokens,
                seq_lens=seq_lens,
                query_lens=query_lens,
                input_positions=input_positions,
                attn_metadata=prefill_attn_metadata,
                lora_requests=lora_requests,
                lora_mapping=lora_prompt_mapping,
                multi_modal_kwargs=multi_modal_kwargs,
                real_batch_size=real_batch_size,
                batch_size_padded=batch_size_padded,
                lora_ids=lora_ids,
            )

            prompt_offsets = [
                i * model_input.input_tokens.shape[1]
                for i in range(model_input.batch_size_padded)
            ]
            prompt_offsets_tensor = torch.tensor(prompt_offsets).to(
                model_input.input_tokens.device)

            # Prepare PoolingMetadata.
            assert model_input.seq_lens is not None
            pooling_metadata = self._prepare_pooling(
                seq_group_metadata_list,
                model_input.seq_lens,
                prompt_offsets=prompt_offsets_tensor)

        return dataclasses.replace(model_input,
                                   pooling_metadata=pooling_metadata)

    def _prepare_pooling(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        prompt_offsets: List[int],
    ) -> PoolingMetadata:
        """Prepare PoolingMetadata for the sequence group metadata list."""
        seq_groups: List[Tuple[List[int], PoolingParams]] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            pooling_params = seq_group_metadata.pooling_params
            seq_groups.append((seq_ids, pooling_params))

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        pooling_metadata = PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            prompt_offsets=prompt_offsets,
        )

        return pooling_metadata