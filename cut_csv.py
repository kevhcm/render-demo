import argparse
import csv


def cut_first_rows(input_path: str, output_path: str, n: int, *, encoding: str = "utf-8-sig") -> None:
    """
    Copy header + first n data rows from a CSV to output_path.

    This intentionally uses the standard library `csv` module so the demo stays lightweight.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    with open(input_path, "r", newline="", encoding=encoding) as f_in, open(
        output_path, "w", newline="", encoding="utf-8-sig"
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        header = next(reader, None)
        if header is None:
            # Empty CSV: produce empty output.
            return

        writer.writerow(header)

        count = 0
        for row in reader:
            writer.writerow(row)
            count += 1
            if count >= n:
                break


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut first N rows from a CSV (keeps header).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=5, help="Number of data rows to keep (header always kept).")
    args = ap.parse_args()

    cut_first_rows(args.input, args.output, args.n)


if __name__ == "__main__":
    main()

