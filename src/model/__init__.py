"""Model package exports.

Heavy runtime dependencies such as ``torch`` may be absent in lightweight
inspection environments, so optional imports are guarded.
"""

from .registry import list_architectures, register_architecture
from .specs import ModelSpec

__all__ = [
    "ModelSpec",
    "list_architectures",
    "register_architecture",
]

try:
    from .data import (
        TaikoDataset,
        build_chart_manifest,
        build_sequence_index,
        build_vocab_from_all_splits,
        encode_tokens,
        infer_beatmap_id_value,
        infer_density_nps,
        infer_difficulty_value,
        load_one_sample,
        preprocess_beatmap_id,
        preprocess_density_nps,
        preprocess_difficulty_value,
        split_chart_manifest,
        taiko_collate_fn,
    )
    from .generation import (
        SamplingConfig,
        TaikoBeatmapGenerator,
        compare_song_output_with_notes_json,
    )
    from .model import (
        AudioEmbedding,
        AudioEncoder,
        ChartDecoder,
        OutputHead,
        PositionalEncoding,
        TaikoTransformer,
        TokenEmbedding,
        TransformerBaselineConfig,
        build_model,
        build_transformer_baseline,
        generate_causal_mask,
    )
    from .trainer import fit, plot_loss, train_one_epoch, validate_one_epoch

    __all__.extend(
        [
            "AudioEmbedding",
            "AudioEncoder",
            "ChartDecoder",
            "OutputHead",
            "PositionalEncoding",
            "SamplingConfig",
            "TaikoBeatmapGenerator",
            "TaikoDataset",
            "TaikoTransformer",
            "TokenEmbedding",
            "TransformerBaselineConfig",
            "build_chart_manifest",
            "build_model",
            "build_sequence_index",
            "build_transformer_baseline",
            "build_vocab_from_all_splits",
            "compare_song_output_with_notes_json",
            "encode_tokens",
            "fit",
            "generate_causal_mask",
            "infer_beatmap_id_value",
            "infer_density_nps",
            "infer_difficulty_value",
            "load_one_sample",
            "plot_loss",
            "preprocess_beatmap_id",
            "preprocess_density_nps",
            "preprocess_difficulty_value",
            "split_chart_manifest",
            "taiko_collate_fn",
            "train_one_epoch",
            "validate_one_epoch",
        ]
    )
except ModuleNotFoundError:
    pass
