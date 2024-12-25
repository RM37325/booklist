import json
import pandas as pd
import argparse

def convert_json_to_csv(input_json_path, output_csv_path):
    # JSONファイルを読み込む
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # "evaluations"セクションをデータフレームに変換
    evaluations = data['evaluations']

    # 必要なデータを抽出して整形
    df = pd.DataFrame([
        {
            "text_id": eval_item["text_id"],
            "text": eval_item["text"],
            "primary_sentiment": eval_item["primary_sentiment"],
            "strength": eval_item["strength"],
            "positive_score": eval_item["scores"]["positive"],
            "neutral_score": eval_item["scores"]["neutral"],
            "negative_score": eval_item["scores"]["negative"],
            "used_models": ", ".join(eval_item["used_models"]),
            "all_models_used": eval_item["all_models_used"]
        }
        for eval_item in evaluations
    ])

    # データフレームをCSVファイルとして保存
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"データがCSVファイルとして保存されました: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Drive上のJSONファイルをCSVに変換するスクリプト")
    parser.add_argument("input_json", help="Google Drive上の入力JSONファイルのパス")
    parser.add_argument("output_csv", help="Google Drive上の出力CSVファイルのパス")

    args = parser.parse_args()
    convert_json_to_csv(args.input_json, args.output_csv)
