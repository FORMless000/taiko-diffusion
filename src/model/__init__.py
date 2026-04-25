from .data import (
    build_chart_manifest,
    split_chart_manifest,
    build_sequence_index,
    load_one_sample,
    build_vocab_from_all_splits,
    encode_tokens,
    infer_difficulty_value,
    infer_density_nps,
    infer_beatmap_id_value,
    preprocess_difficulty_value,
    preprocess_density_nps,
    preprocess_beatmap_id,
    TaikoDataset,
    TaikoContextDataset,
    taiko_collate_fn,
    taiko_context_collate_fn,
    build_dataset_for_spec,
)
from .specs import (
    ArchitectureSpec,
    TrainingSpec,
)
from .factory import (
    build_model,
    register_model_builder,
)
from .checkpoints import (
    CheckpointMetadata,
    capture_rng_states,
    diffusion_refiner_architecture_spec,
    export_diffusion_inference_bundle,
    normalize_vocab_payload,
    restore_rng_states,
    save_checkpoint,
    save_inference_bundle,
    load_checkpoint,
    load_inference_artifacts,
)
from .train_api import (
    TrainingArtifacts,
    DatasetBundle,
    TrainingContext,
    build_training_artifacts,
    prepare_sample_data_artifacts,
    create_dataset_bundle,
    create_training_context,
    load_training_context_from_checkpoint,
    train_context,
)
from .model import (
    AudioEmbedding,
    PositionalEncoding,
    AudioEncoder,
    TokenEmbedding,
    ChartDecoder,
    generate_causal_mask,
    OutputHead,
    TaikoTransformer,
)
from .diffusion_refiner import (
    TaikoDiffusionRefiner,
)
from .taiko_context import (
    TaikoContextTransformer,
)
from .trainer import (
    train_one_epoch,
    validate_one_epoch,
    fit,
    plot_loss,
)
from .wandb_utils import (
    WandbConfig,
    WandbRuntime,
    setup_wandb_runtime,
)
from .runtime import (
    PrecisionRuntime,
    normalize_precision,
    resolve_precision_runtime,
    build_grad_scaler,
    build_dataloader_runtime_kwargs,
)

try:
    from .generation import (
        SamplingConfig,
        TaikoBeatmapGenerator,
        compare_song_output_with_notes_json,
    )
except ImportError:
    SamplingConfig = None
    TaikoBeatmapGenerator = None
    compare_song_output_with_notes_json = None

__all__ = [
    "build_chart_manifest",
    "split_chart_manifest",
    "build_sequence_index",
    "load_one_sample",
    "build_vocab_from_all_splits",
    "encode_tokens",
    "infer_difficulty_value",
    "infer_density_nps",
    "infer_beatmap_id_value",
    "preprocess_difficulty_value",
    "preprocess_density_nps",
    "preprocess_beatmap_id",
    "TaikoDataset",
    "TaikoContextDataset",
    "taiko_collate_fn",
    "taiko_context_collate_fn",
    "build_dataset_for_spec",
    "ArchitectureSpec",
    "TrainingSpec",
    "build_model",
    "register_model_builder",
    "CheckpointMetadata",
    "capture_rng_states",
    "diffusion_refiner_architecture_spec",
    "export_diffusion_inference_bundle",
    "normalize_vocab_payload",
    "restore_rng_states",
    "save_checkpoint",
    "save_inference_bundle",
    "load_checkpoint",
    "load_inference_artifacts",
    "TrainingArtifacts",
    "DatasetBundle",
    "TrainingContext",
    "build_training_artifacts",
    "prepare_sample_data_artifacts",
    "create_dataset_bundle",
    "create_training_context",
    "load_training_context_from_checkpoint",
    "train_context",
    "AudioEmbedding",
    "PositionalEncoding",
    "AudioEncoder",
    "TokenEmbedding",
    "ChartDecoder",
    "generate_causal_mask",
    "OutputHead",
    "TaikoTransformer",
    "TaikoDiffusionRefiner",
    "TaikoContextTransformer",
    "train_one_epoch",
    "validate_one_epoch",
    "fit",
    "plot_loss",
    "WandbConfig",
    "WandbRuntime",
    "setup_wandb_runtime",
    "PrecisionRuntime",
    "normalize_precision",
    "resolve_precision_runtime",
    "build_grad_scaler",
    "build_dataloader_runtime_kwargs",
]

if SamplingConfig is not None:
    __all__.extend(
        [
            "SamplingConfig",
            "TaikoBeatmapGenerator",
            "compare_song_output_with_notes_json",
        ]
    )
