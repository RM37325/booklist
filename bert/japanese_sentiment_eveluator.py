import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデル名のリスト
models = {
    "bert_japanese_sentiment": {
        "model_name": "LoneWolfgang/bert-for-japanese-twitter-sentiment",
    },
    "koheiduck_bert_japanese_sentiment": {
        "model_name": "koheiduck/bert-japanese-finetuned-sentiment",
    },
    "bert_finetuned_japanese_sentiment": {
        "model_name": "christian-phu/bert-finetuned-japanese-sentiment",
    },
}


def load_model_and_tokenizer(model_name):
    # モデルとトークナイザーを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


def load_texts_from_json(filepath):
    # JSONファイルからテキストリストを読み込む
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            texts = []
            for item in json_data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                else:
                    print(f"Warning: Skipping invalid item format: {item}")
            return texts
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def preprocess_text(texts):
    # テキストの前処理を行う
    parsed_texts = []
    for text_idx, text in enumerate(texts):
        lines = []
        for txt in text.split("\n"):
            txt = txt.strip()
            lines.append(txt.strip())
        parsed_texts.append("".join(lines))
    return parsed_texts


def predict_sentiment(model_key, tokenizer, model, texts):
    # 感情分析の予測を実行
    model_results = {}

    for text_idx, text in enumerate(texts):
        lines = []
        for txt in text.split("\n"):
            lines.append(txt.strip())

        batch_size = 32
        line_results = []

        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i : i + batch_size]
            inputs = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            label_mapping = model.config.id2label

            for idx_in_batch, line in enumerate(batch_lines):
                scores = predictions[idx_in_batch].tolist()
                results = [
                    {"label": label_mapping.get(i, f"Label_{i}"), "score": score}
                    for i, score in enumerate(scores)
                ]
                line_results.append({"line": line, "results": results})

        model_results[text_idx] = line_results

    return model_results


def summarize_sentiments(all_evaluations):
    # 全テキストの感情分析結果を集計
    total_count = len(all_evaluations)

    # 基本的な感情の集計
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    # 感情の強度も含めた詳細な集計
    detailed_counts = {
        "positive": {"Strong": 0, "Moderate": 0, "Weak": 0},
        "neutral": {"Strong": 0, "Moderate": 0, "Weak": 0},
        "negative": {"Strong": 0, "Moderate": 0, "Weak": 0},
    }

    for eval in all_evaluations:
        sentiment = eval["primary_sentiment"]
        strength = eval["strength"]

        # 基本的な感情のカウント
        sentiment_counts[sentiment] += 1

        # 強度を含めた詳細なカウント
        detailed_counts[sentiment][strength] += 1

    # パーセンテージの計算
    sentiment_percentages = {
        sentiment: (count / total_count) * 100
        for sentiment, count in sentiment_counts.items()
    }

    detailed_percentages = {
        sentiment: {
            strength: (count / total_count) * 100
            for strength, count in strengths.items()
        }
        for sentiment, strengths in detailed_counts.items()
    }

    # 結果の表示
    print("\n=== Sentiment Analysis Summary ===")
    print(f"\nTotal texts analyzed: {total_count}")

    print("\nBasic Sentiment Distribution:")
    for sentiment in ["positive", "neutral", "negative"]:
        count = sentiment_counts[sentiment]
        percentage = sentiment_percentages[sentiment]
        print(f"{sentiment.capitalize():8}: {count:3d} ({percentage:5.1f}%)")

    print("\nDetailed Sentiment Distribution:")
    for sentiment, strengths in detailed_counts.items():
        print(f"\n{sentiment.capitalize()} breakdown:")
        for strength in ["Strong", "Moderate", "Weak"]:
            count = strengths[strength]
            if count > 0:  # 0件の場合は表示しない
                percentage = detailed_percentages[sentiment][strength]
                print(f"  {strength:8}: {count:3d} ({percentage:5.1f}%)")

    return {
        "total": total_count,
        "sentiment_counts": {
            "raw": sentiment_counts,
            "percentages": {k: round(v, 1) for k, v in sentiment_percentages.items()},
        },
        "detailed_counts": {
            "raw": detailed_counts,
            "percentages": {
                sentiment: {
                    strength: round(percentage, 1)
                    for strength, percentage in strengths.items()
                    if detailed_counts[sentiment][strength] > 0
                }
                for sentiment, strengths in detailed_percentages.items()
            },
        },
    }


def evaluate_sentiment(scores):
    # 感情分析スコアを評価して最終的な判定を行う
    # 各感情の閾値を設定
    STRONG_THRESHOLD = 0.7  # 強い感情を示す閾値
    MEDIUM_THRESHOLD = 0.5  # 中程度の感情を示す閾値
    SCORE_DIFFERENCE_THRESHOLD = 0.15  # スコアの差の閾値
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # 高確信度の閾値

    # 各モデルの最大スコアとそのラベルを取得
    model_max_scores = []
    for score in scores:
        max_label = max(["positive", "neutral", "negative"], key=lambda x: score[x])
        model_max_scores.append(
            {
                "model": score["model"],
                "label": max_label,
                "score": score[max_label],
                "scores": score,
            }
        )

    # スコアで降順にソート
    sorted_models = sorted(model_max_scores, key=lambda x: x["score"], reverse=True)

    # 判定が分かれているかチェック
    high_confidence_predictions = [
        m for m in model_max_scores if m["score"] >= HIGH_CONFIDENCE_THRESHOLD
    ]

    different_labels = len(set(m["label"] for m in high_confidence_predictions)) > 1

    # 高確信度で判定が分かれている場合は全モデルを使用
    if different_labels and len(high_confidence_predictions) >= 2:
        filtered_scores = scores
        use_all_models = True
    else:
        # 通常通り上位2モデルを使用
        filtered_scores = [
            s for s in scores if s["model"] in [m["model"] for m in sorted_models[:2]]
        ]
        use_all_models = False

    # スコアの平均を計算
    avg_positive = np.mean([s["positive"] for s in filtered_scores])
    avg_neutral = np.mean([s["neutral"] for s in filtered_scores])
    avg_negative = np.mean([s["negative"] for s in filtered_scores])

    sentiment_scores = {
        "positive": avg_positive,
        "neutral": avg_neutral,
        "negative": avg_negative,
    }

    # スコアの差を確認
    sorted_scores = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    score_difference = sorted_scores[0][1] - sorted_scores[1][1]

    # 最も高いスコアを持つ感情を特定
    primary_sentiment = sorted_scores[0][0]

    # スコアが近接している場合はneutralとして判定
    if score_difference < SCORE_DIFFERENCE_THRESHOLD:
        primary_sentiment = "neutral"
        strength = "Weak"
    else:
        if sorted_scores[0][1] >= STRONG_THRESHOLD:
            strength = "Strong"
        elif sorted_scores[0][1] >= MEDIUM_THRESHOLD:
            strength = "Moderate"
        else:
            strength = "Weak"

    used_models = [
        m["model"] for m in sorted_models[: 2 if not use_all_models else len(scores)]
    ]

    return {
        "primary_sentiment": primary_sentiment,
        "strength": strength,
        "scores": sentiment_scores,
        "used_models": used_models,
        "all_models_used": use_all_models,
    }


def ensure_directory_exists(file_path):
    # ファイルのディレクトリが存在することを確認し、必要に応じて作成
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(args):
    # 入力ファイルの存在確認
    input_path = Path(args.jsonfile)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # 出力ファイルパスの設定
    output_path = input_path.parent / f"{input_path.stem}_analysis.json"

    # 出力ディレクトリの確認/作成
    ensure_directory_exists(output_path)

    # テキストの読み込み
    print(f"Loading texts from: {input_path}")
    loaded_texts = load_texts_from_json(input_path)
    texts = preprocess_text(loaded_texts)
    print(f"Loaded {len(texts)} texts")

    # 全てのモデルの結果を格納する辞書
    all_model_results = {}

    # 全ての評価結果を保存
    all_evaluations = []

    # モデルごとのテスト実行
    for model_key, model_info in models.items():
        print(f"Loading model: {model_key}")
        model_name = model_info["model_name"]
        try:
            tokenizer, model = load_model_and_tokenizer(model_name)
            model_results = predict_sentiment(model_key, tokenizer, model, texts)
            all_model_results[model_key] = model_results
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    # 結果の集計と表示
    for text_idx in range(len(texts)):
        print(f"\n---\nText {text_idx + 1}:\n")
        text_content = texts[text_idx]
        print(f"Text content:\n{text_content}\n")

        model_scores = []

        for model_key in models.keys():
            if model_key not in all_model_results:
                continue

            line_results = all_model_results[model_key][text_idx]
            scores_by_label = {"positive": [], "neutral": [], "negative": []}

            for line_result in line_results:
                results = line_result["results"]
                for res in results:
                    label = res["label"].lower()
                    if label in scores_by_label:
                        scores_by_label[label].append(res["score"])

            model_scores.append(
                {
                    "model": model_key,
                    "positive": scores_by_label["positive"][0],
                    "neutral": scores_by_label["neutral"][0],
                    "negative": scores_by_label["negative"][0],
                }
            )

            print(f"\nModel: {model_key}")
            for label in ["positive", "neutral", "negative"]:
                score = scores_by_label[label][0]
                print(f"  {label.capitalize()}: {score:.4f}")

        # 総合評価
        final_evaluation = evaluate_sentiment(model_scores)
        print("\nFinal Evaluation:")
        print(
            f"Primary Sentiment: {final_evaluation['primary_sentiment'].capitalize()}"
        )
        print(f"Strength: {final_evaluation['strength']}")
        print(f"Used Models: {', '.join(final_evaluation['used_models'])}")
        print(
            f"Evaluation Method: {'All models' if final_evaluation['all_models_used'] else 'Top 2 models'}"
        )
        print("\nAggregated Scores:")
        for label, score in final_evaluation["scores"].items():
            print(f"  {label.capitalize()}: {score:.4f}")

        # 評価結果の保存
        final_evaluation.update({"text": text_content, "text_id": text_idx})
        all_evaluations.append(final_evaluation)

    summary = summarize_sentiments(all_evaluations)

    # 結果の保存
    metadata = {
        "total_texts": len(texts),
        "models_used": list(models.keys()),
    }
    try:
        output_data = {
            "metadata": metadata,
            "summary": summary,
            "evaluations": all_evaluations,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment analysis for Japanese texts"
    )
    parser.add_argument(
        "--jsonfile",
        required=True,
        help="Input JSON file path containing texts to analyze",
    )
    args = parser.parse_args()
    main(args)