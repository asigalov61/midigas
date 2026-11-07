#!/usr/bin/env python3
"""
MIDI GAS
---------

Models repo
----------
https://huggingface.co/projectlosangeles/midigas

License
-------
Apache 2.0

Project Los Angeles
Tegridy Code 2026
"""

from __future__ import annotations

#=========================================================================================

from collections import Counter, defaultdict
from statistics import mean, median
from math import sqrt, isfinite

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from x_transformers import TransformerWrapper, Encoder

except ImportError:
    print('x-transformers is not installed!!!')
    print('pip install -U x-transformers==2.31.10')

try:
    from . import TMIDIX

except ImportError:
    print('No TMIDIX found!')


#=========================================================================================

primary_genre_to_id = {
    'alternative': 0,
    'anime piano': 1,
    'blues': 2,
    'chinese piano pop': 3,
    'classical': 4,
    'country': 5,
    'electronic': 6,
    'folk': 7,
    'funk': 8,
    'gospel': 9,
    'hip-hop': 10,
    'jazz': 11,
    'latin': 12,
    'metal': 13,
    'new age': 14,
    'piano pop rock': 15,
    'pop': 16,
    'punk': 17,
    'r&b': 18,
    'reggae': 19,
    'rock': 20,
    'soundtrack': 21,
    'world': 22
}

nuanced_genre_to_id = {
  'alternative': 0,
  'alternative dance': 1,
  'alternative metal': 2,
  'alternative rock': 3,
  'americana': 4,
  'anime piano': 5,
  'art rock': 6,
  'ballad': 7,
  'beat': 8,
  'belgium': 9,
  'black metal': 10,
  'blue-eyed soul': 11,
  'bluegrass': 12,
  'blues': 13,
  'blues-rock': 14,
  'bossa nova': 15,
  'brazilian pop music': 16,
  'brill building pop': 17,
  'british invasion': 18,
  'british pop': 19,
  'cabaret': 20,
  'cantonese pop': 21,
  'ccm': 22,
  'celtic': 23,
  'celtic rock': 24,
  'chanson': 25,
  'chicago soul': 26,
  'chill-out': 27,
  'chinese': 28,
  'chinese music': 29,
  'chinese piano pop': 30,
  'choral music': 31,
  'classic country': 32,
  'classic rock': 33,
  'classical': 34,
  'classical piano strings': 35,
  'club': 36,
  'comedy rock': 37,
  'cool jazz': 38,
  'country': 39,
  'country blues': 40,
  'country gospel': 41,
  'country rock': 42,
  'cumbia': 43,
  'dance': 44,
  'dance pop': 45,
  'dance rock': 46,
  'dark wave': 47,
  'deep house': 48,
  'dirty south rap': 49,
  'disco': 50,
  'disco house': 51,
  'doo-wop': 52,
  'downtempo': 53,
  'drum track': 54,
  'dubstep': 55,
  'easy listening': 56,
  'electro': 57,
  'electronic': 58,
  'electronica': 59,
  'electropop': 60,
  'emo': 61,
  'eurodance': 62,
  'europop': 63,
  'experimental rock': 64,
  'flamenco': 65,
  'folk': 66,
  'folk rock': 67,
  'folk-pop': 68,
  'freakbeat': 69,
  'free improvisation': 70,
  'free jazz': 71,
  'freestyle': 72,
  'funk': 73,
  'funk metal': 74,
  'funky house': 75,
  'future jazz': 76,
  'gabba': 77,
  'gangster rap': 78,
  'garage rock': 79,
  'german pop': 80,
  'glam rock': 81,
  'gospel': 82,
  'grunge': 83,
  'happy hardcore': 84,
  'hard house': 85,
  'hard rock': 86,
  'hard trance': 87,
  'heartland rock': 88,
  'heavy metal': 89,
  'hip hop': 90,
  'hip house': 91,
  'hip pop': 92,
  'hip-hop': 93,
  'honky tonk': 94,
  'indie': 95,
  'indie rock': 96,
  'industrial metal': 97,
  'instrumental pop': 98,
  'instrumental rock': 99,
  'irish folk': 100,
  'israeli': 101,
  'italian disco': 102,
  'j pop': 103,
  'jam band': 104,
  'jangle pop': 105,
  'jazz': 106,
  'jazz funk': 107,
  'jazz piano': 108,
  'jungle music': 109,
  'klezmer': 110,
  'latin': 111,
  'latin jazz': 112,
  'latin pop': 113,
  'los angeles': 114,
  'lullaby': 115,
  'male vocalist': 116,
  'melodies': 117,
  'memphis soul': 118,
  'metal': 119,
  'modern rock': 120,
  'motown': 121,
  'neo classical': 122,
  'neo soul': 123,
  'neo-progressive': 124,
  'neofolk': 125,
  'new age': 126,
  'new romantic': 127,
  'new wave': 128,
  'oldies': 129,
  'opera': 130,
  'orchestra': 131,
  'outlaw country': 132,
  'parody': 133,
  'patriotic': 134,
  'piano pop rock': 135,
  'piano rock': 136,
  'polka': 137,
  'pop': 138,
  'pop folk': 139,
  'pop punk': 140,
  'pop rap': 141,
  'pop rock': 142,
  'post-grunge': 143,
  'power metal': 144,
  'power pop': 145,
  'progressive bluegrass': 146,
  'progressive house': 147,
  'progressive metal': 148,
  'progressive trance': 149,
  'psychedelic rock': 150,
  'punk': 151,
  'quiet storm': 152,
  'r&b': 153,
  'ragtime': 154,
  'ranchera': 155,
  'rap metal': 156,
  'reggae': 157,
  'reggaeton': 158,
  'rock': 159,
  "rock 'n roll": 160,
  'rockabilly': 161,
  'roots reggae': 162,
  'roots rock': 163,
  'russian': 164,
  'samba': 165,
  'show tunes': 166,
  'singer-songwriter': 167,
  'ska': 168,
  'smooth jazz': 169,
  'soft rock': 170,
  'soul jazz': 171,
  'soundtrack': 172,
  'southern rock': 173,
  'soviet children songs': 174,
  'speed metal': 175,
  'speedcore': 176,
  'surf music': 177,
  'sympho black metal': 178,
  'symphonic rock': 179,
  'symphony': 180,
  'synthpop': 181,
  'tech house': 182,
  'techno': 183,
  'teen pop': 184,
  'tejano': 185,
  'trance': 186,
  'tribal house': 187,
  'trip hop': 188,
  'uk garage': 189,
  'unknown': 190,
  'video games music': 191,
  'vocal jazz': 192
}

id_to_primary_genre = {v: k for k, v in primary_genre_to_id.items()}
id_to_nuanced_genre = {v: k for k, v in nuanced_genre_to_id.items()}

#=========================================================================================

# ----------------------------------------------------------------------------
# Dataset & Collate
# ----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    """
    Dataset for pairs (src_seq, label).
    src_seq: list of token IDs (ints).
    label: single int or float (0 or 1).
    """
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_seq = self.data_pairs[idx]
        x = torch.tensor(src_seq, dtype=torch.long)
        return x



def make_collate_fn(pad_idx):
    def collate_fn(batch):
        xs, ys = zip(*batch)
        lengths = [x.size(0) for x in xs]
        max_len = max(lengths)

        x     = torch.full((len(xs), max_len), pad_idx, dtype=torch.long)
        mask  = torch.zeros((len(xs), max_len), dtype=torch.bool)

        for i, (seq, l) in enumerate(zip(xs, lengths)):
            x[i, :l]    = seq
            mask[i, :l] = True

        y = torch.stack(ys).view(-1)            # [B]
        return x, y, mask
    return collate_fn


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(device, vocab_size=1104, num_classes=193, max_seq_len=2048, pad_idx=1103):
    """
    Transformer encoder -> average pooled -> 4 logits (multi-class).
    """
    model = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        logits_dim=num_classes,
        use_cls_token=False,
        average_pool_embed=True,
        emb_dropout = 0.1,
        attn_layers=Encoder(
            dim=512,
            depth=8,
            heads=8,
            pre_and_post_norm=True,
            rotary_pos_emb=False,
            polar_pos_emb=True,
            attn_flash=True,
            layer_dropout = 0.1,   # stochastic depth - dropout entire layer
            attn_dropout = 0.1,    # dropout post-attention
            ff_dropout = 0.1       # feedforward dropout
        ),
    )
    return model.to(device)

def load_model(checkpoint_path, num_classes=193, device='cuda'):
    """
    Rebuilds the architecture, loads weights.
    """
    model = build_model(device, num_classes=num_classes)                 # from your original code
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

@torch.no_grad()
def predict(model, seqs, device='cuda', batch_size=8, pad_idx=992):
    """
    Returns two lists:
      - preds: int class predictions (0, 1, 2, 3)
      - probs: float probabilities for the predicted class
    """
    seqs_lens = [len(s) for s in seqs]
    
    device = torch.device(device)
    
    model.to(device)
    model.eval()
    
    # --- 1. Pad sequences and build masks ---
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 1
    
    x = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    
    for i, (seq, l) in enumerate(zip(seqs, lengths)):
        x[i, :l] = torch.tensor(seq, dtype=torch.long)
        mask[i, :l] = True

    all_preds = []
    all_probs = []

    # --- 2. Batched Inference ---
    for i in range(0, len(seqs), batch_size):
        batch_x = x[i:i+batch_size].to(device, non_blocking=True)
        batch_mask = mask[i:i+batch_size].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch_x, mask=batch_mask)  # [B, 4]

        # Softmax to get probabilities, then pick the max
        probs = torch.softmax(logits.float(), dim=-1)  # [B, 4]
        confidences, preds = probs.max(dim=-1)         # [B], [B]

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(confidences.cpu().tolist())

    return all_preds, all_probs, seqs_lens

def classification_report(preds, probs, seq_lens, id_to_genre, print_report=True):
    """
    Generate a detailed MIDI genre classification report.

    Includes:
        - Dominant genre prediction
        - Prediction consensus ranking
        - Confidence ranking (mean confidence per genre)
        - Per-genre confidence statistics
        - Overall confidence statistics
        - Sequence length statistics

    Returns:
        dict: Structured classification statistics.
    """

    preds = list(preds)
    probs = [float(p) for p in probs]
    seq_lens = list(seq_lens)

    if not preds:
        raise ValueError("preds cannot be empty.")

    if not (len(preds) == len(probs) == len(seq_lens)):
        raise ValueError(
            "preds, probs, and seq_lens must have identical lengths."
        )

    if any(not isfinite(p) or not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError("Probabilities must be finite and between 0 and 1.")

    if any(n < 0 for n in seq_lens):
        raise ValueError("Sequence lengths cannot be negative.")

    # --------------------------------------------------
    # Prediction statistics
    # --------------------------------------------------

    counts = Counter(preds)

    dominant_id = counts.most_common(1)[0][0]
    dominant_count = counts[dominant_id]

    def get_genre(class_id):
        return id_to_genre.get(class_id, f"Unknown ({class_id})")

    genre = get_genre(dominant_id)

    # --------------------------------------------------
    # Per-genre confidence statistics
    # --------------------------------------------------

    genre_probs = defaultdict(list)

    for pred, prob in zip(preds, probs):
        genre_probs[pred].append(prob)

    genre_statistics = {}

    for class_id, values in genre_probs.items():
        genre_statistics[class_id] = {
            "genre": get_genre(class_id),
            "count": len(values),
            "percentage": len(values) / len(preds) * 100,
            "mean_confidence": mean(values),
            "median_confidence": median(values),
            "min_confidence": min(values),
            "max_confidence": max(values),
            "total_confidence": sum(values),
        }

    # --------------------------------------------------
    # Ranking 1: Prediction consensus
    # --------------------------------------------------

    consensus_ranking = sorted(
        genre_statistics.items(),
        key=lambda item: (
            -item[1]["count"],
            -item[1]["mean_confidence"]
        )
    )

    # --------------------------------------------------
    # Ranking 2: Mean confidence
    # --------------------------------------------------

    confidence_ranking = sorted(
        genre_statistics.items(),
        key=lambda item: (
            -item[1]["mean_confidence"],
            -item[1]["count"]
        )
    )

    # --------------------------------------------------
    # Overall confidence statistics
    # --------------------------------------------------

    avg_conf = mean(probs)
    median_conf = median(probs)

    std_conf = sqrt(
        mean((p - avg_conf) ** 2 for p in probs)
    )

    dominant_probs = genre_probs[dominant_id]

    confidence = {
        "mean": avg_conf,
        "median": median_conf,
        "min": min(probs),
        "max": max(probs),
        "std": std_conf,
        "dominant_class_mean": mean(dominant_probs),
    }

    # --------------------------------------------------
    # Sequence statistics
    # --------------------------------------------------

    sequence = {
        "total_length": sum(seq_lens),
        "mean_length": mean(seq_lens),
        "median_length": median(seq_lens),
        "min_length": min(seq_lens),
        "max_length": max(seq_lens),
    }

    # --------------------------------------------------
    # Structured report
    # --------------------------------------------------

    report = {
        "genre": genre,
        "genre_id": dominant_id,
        "num_predictions": len(preds),
        "prediction_consistency": dominant_count / len(preds) * 100,
        "confidence": confidence,
        "sequence": sequence,
        "distribution": {
            class_id: {
                "genre": stats["genre"],
                "count": stats["count"],
                "percentage": stats["percentage"],
            }
            for class_id, stats in genre_statistics.items()
        },
        "genre_statistics": genre_statistics,
        "consensus_ranking": [
            {"genre_id": cid, **stats}
            for cid, stats in consensus_ranking
        ],
        "confidence_ranking": [
            {"genre_id": cid, **stats}
            for cid, stats in confidence_ranking
        ],
    }

    # --------------------------------------------------
    # Formatted terminal report
    # --------------------------------------------------

    if print_report:

        width = 58
        line = "=" * width

        def pct(value):
            return f"{value * 100:6.2f}%"

        def heading(title):
            print(f"\n{title:^58}")
            print("  " + "-" * 54)

        print()
        print(line)
        print("              CLASSIFICATION REPORT")
        print(line)

        print(f"\n  Predicted Genre     : {genre}")
        print(f"  Class ID            : {dominant_id}")
        print(f"  Predictions         : {len(preds)}")
        print(
            f"  Prediction Agreement: "
            f"{pct(dominant_count / len(preds))}"
        )

        # Confidence
        heading("CONFIDENCE STATISTICS")

        print(f"  Mean Confidence     : {pct(avg_conf)}")
        print(f"  Median Confidence   : {pct(median_conf)}")
        print(f"  Dominant Class Mean : {pct(mean(dominant_probs))}")
        print(f"  Minimum Confidence  : {pct(min(probs))}")
        print(f"  Maximum Confidence  : {pct(max(probs))}")
        print(f"  Std. Deviation      : {pct(std_conf)}")

        # Sequence
        heading("SEQUENCE STATISTICS")

        print(f"  Total Length        : {sequence['total_length']:,}")
        print(f"  Average Length      : {sequence['mean_length']:,.2f}")
        print(f"  Median Length       : {sequence['median_length']:,.2f}")
        print(f"  Minimum Length      : {sequence['min_length']:,}")
        print(f"  Maximum Length      : {sequence['max_length']:,}")

        # Consensus ranking
        heading("GENRE CONSENSUS RANKING")

        for rank, (class_id, stats) in enumerate(
            consensus_ranking, 1
        ):
            bar = "#" * round(stats["percentage"] / 2)

            print(
                f"  {rank:>2}. {stats['genre']:<20} "
                f"{stats['count']:>3} "
                f"({stats['percentage']:5.1f}%) {bar}"
            )

        # Confidence ranking
        heading("GENRE CONFIDENCE RANKING")

        for rank, (class_id, stats) in enumerate(
            confidence_ranking, 1
        ):
            print(
                f"  {rank:>2}. {stats['genre']:<20} "
                f"{pct(stats['mean_confidence'])} "
                f"| n={stats['count']:>3} "
                f"| total={stats['total_confidence']:.3f}"
            )

        print("\n" + line)
        print()

    return report

def midi_to_tokens(midi_file_path, chunk_len=2048, chunk_overlap=512):

    try:

        raw_score = TMIDIX.midi2single_track_ms_score(
            midi_file_path, do_not_check_MIDI_signature=True
        )
        
        escore = TMIDIX.advanced_score_processor(
            raw_score, return_enhanced_score_notes=True, apply_sustain=True
        )
    
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore[0], sort_drums_last=True)
    
        escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)
        
        escore_notes = TMIDIX.fix_escore_notes_durations(escore_notes, min_notes_gap=0)
    
        tscore = TMIDIX.transpose_escore_notes_to_pitch(escore_notes)
                
        cscore = TMIDIX.chordify_score([1000, tscore])
    
        fixed_score = []
    
        for c in cscore:
            c.sort(key=lambda x: -x[4])
    
            tones_chord = sorted(set([p[4] % 12 for p in c if p[3] != 9]))
    
            if tones_chord:
                if tones_chord not in TMIDIX.ALL_CHORDS_SORTED:
                    tones_chord = TMIDIX.check_and_fix_tones_chord(tones_chord, use_full_chords=False)
    
                for e in c:
                    if e[3] != 9:
                        if e[4] % 12 in tones_chord:
                            fixed_score.append(e)
    
                    else:
                        fixed_score.append(e)
    
            else:
                fixed_score.extend(c)
    
        cscore = TMIDIX.chordify_score([1000, fixed_score])
    
        score = []
        bar_time_ms = 4096
        dtime = 0
        abs_time = 0
        pbar = -1
        bar_div = max(1, max(1, bar_time_ms) // 16)
    
        pc = cscore[0]
    
        for c in cscore:
            c.sort(key=lambda x: -x[4])
            
            dtime = max(0, min(255, c[0][1]-pc[0][1]))
            abs_time += dtime
            
            if abs_time // bar_div > pbar:
                score.append(718) # Single bar token
                pbar = abs_time // bar_div
    
            tones_chord = sorted(set([p[4] % 12 for p in c if p[3] != 9]))
    
            if tones_chord:
                if len(tones_chord) > 1:
                    chord_tok = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)+12
        
                else:
                    chord_tok = tones_chord[0]
    
            else:
                chord_tok = 333
    
            score.append(chord_tok+640)
            
            score.append(dtime)
    
            for e in c:
                score.append(max(0, min(128, e[6]))+974) # instruments
                score.extend([max(1, min(127, e[4]))+256, max(1, min(255, e[2]))+384]) # pitch / dur
    
            pc = c
    
        return TMIDIX.chunk_by_size(score, chunk_len, chunk_overlap)

    except:
        return []