"""Nhận dạng một file wav bằng KNN đã huấn luyện."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recognize import predict_path
from src.config import configure_console

configure_console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    args = parser.parse_args()
    result = predict_path(Path(args.wav))
    if result["rejected"]:
        print("Không chắc (từ ngoài danh sách hoặc nhiễu).")
    else:
        print(f"{result['word_text']}  ({result['word_code']})  tin cậy={result['confidence']}")
    print(result)


if __name__ == "__main__":
    main()
