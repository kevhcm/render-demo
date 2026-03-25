import io
import os
import tempfile

from flask import Flask, jsonify, request, Response
from werkzeug.utils import secure_filename

from cut_csv import cut_first_rows

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> Response:
    # Render often points users at the root URL; provide a friendly landing page.
    return Response(
        "CSV Cut Demo. Use POST /cut with multipart form-data fields: file=<csv> and optional n=<int>.\n",
        mimetype="text/plain",
    )


@app.route("/health", methods=["GET"])
def health() -> Response:
    return Response("ok\n", mimetype="text/plain")


@app.route("/cut", methods=["POST"])
def cut() -> Response:
    """
    Accept multipart upload:
      - file: CSV file
      - n: optional number of data rows to keep (default=5)

    Returns the cut CSV as an attachment.
    """
    if "file" not in request.files:
        return jsonify({"error": "missing form field: file"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "empty filename"}), 400

    n_raw = request.form.get("n", "").strip()
    try:
        n = int(n_raw) if n_raw else 5
    except ValueError:
        return jsonify({"error": "n must be an integer"}), 400

    # Render will mount us with only the app directory; use temp storage per request.
    filename = secure_filename(uploaded.filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input_{filename}")
        output_path = os.path.join(tmpdir, "output.csv")

        uploaded.save(input_path)
        try:
            cut_first_rows(input_path, output_path, n)
        except Exception as e:
            return jsonify({"error": f"failed to cut csv: {str(e)}"}), 500

        with open(output_path, "rb") as f:
            csv_bytes = f.read()

    headers = {"Content-Disposition": 'attachment; filename="forecast_output.csv"'}
    return Response(csv_bytes, mimetype="text/csv", headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

