import os, shutil, tempfile, subprocess, uuid
from pathlib import Path
from flask import Flask, request, send_file, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

UPLOAD_DIR = Path(tempfile.gettempdir()) / "label_cropper"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Core crop logic ────────────────────────────────────────────────────────────
def crop_flipkart_labels(input_pdf: str, output_pdf: str):
    from PIL import Image
    import numpy as np
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    DPI       = 200
    margin_px = int(0.3 / 2.54 * DPI)

    tmp_dir = tempfile.mkdtemp()
    prefix  = os.path.join(tmp_dir, "page")

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm not found on server.")

    result = subprocess.run(
        [pdftoppm, "-r", str(DPI), input_pdf, prefix],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Render failed: {result.stderr}")

    pages = sorted(Path(tmp_dir).glob("page-*.ppm"))
    if not pages:
        raise RuntimeError("No pages found in PDF.")

    cropped_paths = []
    for idx, ppm_path in enumerate(pages):
        img  = Image.open(str(ppm_path)).convert("RGB")
        arr  = __import__('numpy').array(img)
        H, W = arr.shape[:2]

        MIN_COL = max(300, int(H * 0.12))
        MIN_ROW = max(200, int(W * 0.12))

        label_left  = next((x for x in range(W)     if __import__('numpy').sum(arr[:,x,:].min(axis=1)<100) > MIN_COL), 0)
        label_right = next((x for x in range(W-1,0,-1) if __import__('numpy').sum(arr[:,x,:].min(axis=1)<100) > MIN_COL), W-1)
        label_top   = next((y for y in range(H)     if __import__('numpy').sum(arr[y,:,:].min(axis=1)<100) > MIN_ROW), 0)

        span = label_right - label_left
        solid_rows = []
        for y in range(H-1, H//4, -1):
            if __import__('numpy').sum(arr[y, label_left:label_right, :].min(axis=1) < 100) > span * 0.70:
                solid_rows.append(y)
                if len(solid_rows) >= 30: break
        label_bottom = max(solid_rows) if solid_rows else H-1

        arr[label_bottom+2:, :] = 255
        ct = max(0, label_top    - margin_px)
        cl = max(0, label_left   - margin_px)
        cr = min(W, label_right  + margin_px)
        cb = min(H, label_bottom + margin_px)

        cropped  = arr[ct:cb, cl:cr]
        out_img  = Image.fromarray(cropped)
        out_path = os.path.join(tmp_dir, f"crop_{idx+1:03d}.png")
        out_img.save(out_path, dpi=(DPI, DPI))
        cropped_paths.append(out_path)

    c = None
    for img_path in cropped_paths:
        img = Image.open(img_path)
        W_pt = img.size[0] / DPI * 72
        H_pt = img.size[1] / DPI * 72
        if c is None:
            c = rl_canvas.Canvas(output_pdf, pagesize=(W_pt, H_pt))
        else:
            c.setPageSize((W_pt, H_pt))
        c.drawImage(ImageReader(img_path), 0, 0, W_pt, H_pt)
        c.showPage()
    if c:
        c.save()

    shutil.rmtree(tmp_dir, ignore_errors=True)

# ── HTML UI ────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flipkart Label Cropper</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #F0F2FF; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
  .card { background: #fff; border-radius: 20px; box-shadow: 0 8px 40px rgba(91,106,240,0.13); padding: 40px 36px; width: 100%; max-width: 500px; }
  .logo { text-align: center; margin-bottom: 28px; }
  .logo h1 { font-size: 22px; font-weight: 700; color: #1E1E2E; margin-top: 10px; }
  .logo p { color: #6B7280; font-size: 13px; margin-top: 4px; }
  .icon-box { width: 56px; height: 56px; background: #EEF0FD; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 26px; }

  .drop-zone { border: 2.5px dashed #5B6AF0; border-radius: 14px; background: #F5F6FF; padding: 36px 20px; text-align: center; cursor: pointer; transition: all 0.2s; position: relative; margin-bottom: 20px; }
  .drop-zone:hover, .drop-zone.dragover { background: #EEF0FD; border-color: #4454D6; }
  .drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; }
  .drop-zone .dz-icon { font-size: 36px; margin-bottom: 8px; }
  .drop-zone h3 { color: #5B6AF0; font-size: 15px; font-weight: 600; }
  .drop-zone p { color: #9CA3AF; font-size: 12px; margin-top: 4px; }
  .file-name { color: #1E1E2E; font-size: 13px; margin-top: 8px; font-weight: 500; display: none; }

  .btn { width: 100%; padding: 14px; background: #5B6AF0; color: #fff; border: none; border-radius: 12px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; }
  .btn:hover { background: #4454D6; }
  .btn:disabled { background: #A5B4FC; cursor: not-allowed; }

  .progress-wrap { margin: 18px 0 0; display: none; }
  .progress-bar { height: 8px; background: #E5E7EB; border-radius: 99px; overflow: hidden; }
  .progress-fill { height: 100%; background: #5B6AF0; border-radius: 99px; width: 0%; transition: width 0.4s ease; }
  .status { font-size: 13px; color: #6B7280; margin-top: 8px; text-align: center; }

  .result { display: none; margin-top: 20px; background: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 12px; padding: 16px 20px; }
  .result h3 { color: #166534; font-size: 14px; font-weight: 600; margin-bottom: 10px; }
  .dl-btn { display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; padding: 12px; background: #22C55E; color: #fff; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; text-decoration: none; transition: background 0.2s; }
  .dl-btn:hover { background: #16A34A; }
  .another { margin-top: 10px; width: 100%; padding: 10px; background: #fff; color: #5B6AF0; border: 1.5px solid #5B6AF0; border-radius: 10px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .another:hover { background: #EEF0FD; }

  .error { display: none; margin-top: 16px; background: #FEF2F2; border: 1.5px solid #FCA5A5; border-radius: 12px; padding: 14px 18px; color: #991B1B; font-size: 13px; }
  .steps { margin-top: 28px; border-top: 1px solid #F3F4F6; padding-top: 20px; }
  .steps h4 { font-size: 12px; color: #9CA3AF; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 12px; }
  .step { display: flex; gap: 12px; margin-bottom: 10px; align-items: flex-start; }
  .step-num { min-width: 22px; height: 22px; background: #EEF0FD; color: #5B6AF0; border-radius: 50%; font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; }
  .step-txt { font-size: 13px; color: #374151; line-height: 1.5; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <div class="icon-box">📦</div>
    <h1>Flipkart Label Cropper</h1>
    <p>Upload your PDF — get clean shipping labels instantly</p>
  </div>

  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept=".pdf">
    <div class="dz-icon">📄</div>
    <h3>Click to upload PDF</h3>
    <p>or drag and drop here</p>
    <div class="file-name" id="fileName"></div>
  </div>

  <button class="btn" id="cropBtn" disabled onclick="cropPDF()">
    ✂ Crop Labels
  </button>

  <div class="progress-wrap" id="progressWrap">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="status" id="statusMsg">Processing...</div>
  </div>

  <div class="result" id="resultBox">
    <h3>✅ Labels cropped successfully!</h3>
    <a class="dl-btn" id="downloadBtn" href="#" download>⬇ Download Cropped PDF</a>
    <button class="another" onclick="resetApp()">🔄 Crop Another PDF</button>
  </div>

  <div class="error" id="errorBox"></div>

  <div class="steps">
    <h4>How it works</h4>
    <div class="step"><div class="step-num">1</div><div class="step-txt">Upload your Flipkart / E-Kart invoice PDF</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-txt">App auto-detects & crops the shipping label</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-txt">Download the clean PDF — ready to print!</div></div>
  </div>
</div>

<script>
  const fileInput = document.getElementById('fileInput');
  const dropZone  = document.getElementById('dropZone');
  const cropBtn   = document.getElementById('cropBtn');
  const fileName  = document.getElementById('fileName');
  let selectedFile = null;

  fileInput.addEventListener('change', e => setFile(e.target.files[0]));
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragover'); setFile(e.dataTransfer.files[0]); });

  function setFile(f) {
    if (!f || !f.name.endsWith('.pdf')) { alert('Please select a PDF file.'); return; }
    selectedFile = f;
    fileName.textContent = '📄 ' + f.name;
    fileName.style.display = 'block';
    cropBtn.disabled = false;
    document.getElementById('resultBox').style.display = 'none';
    document.getElementById('errorBox').style.display  = 'none';
  }

  function setProgress(pct, msg) {
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('statusMsg').textContent = msg;
  }

  async function cropPDF() {
    if (!selectedFile) return;
    cropBtn.disabled = true;
    cropBtn.textContent = '⏳ Processing...';
    document.getElementById('progressWrap').style.display = 'block';
    document.getElementById('resultBox').style.display  = 'none';
    document.getElementById('errorBox').style.display   = 'none';
    setProgress(10, 'Uploading PDF...');

    const fd = new FormData();
    fd.append('pdf', selectedFile);

    try {
      setProgress(30, 'Rendering pages...');
      const res  = await fetch('/crop', { method: 'POST', body: fd });
      setProgress(80, 'Assembling output...');
      const data = await res.json();

      if (data.error) throw new Error(data.error);

      setProgress(100, 'Done!');
      const dlBtn = document.getElementById('downloadBtn');
      dlBtn.href = '/download/' + data.file_id;
      dlBtn.download = data.filename;
      document.getElementById('resultBox').style.display = 'block';
      document.getElementById('progressWrap').style.display = 'none';

    } catch(err) {
      document.getElementById('errorBox').textContent = '❌ ' + err.message;
      document.getElementById('errorBox').style.display = 'block';
      document.getElementById('progressWrap').style.display = 'none';
    }

    cropBtn.disabled = false;
    cropBtn.innerHTML = '✂ Crop Labels';
  }

  function resetApp() {
    selectedFile = null;
    fileInput.value = '';
    fileName.style.display = 'none';
    cropBtn.disabled = true;
    document.getElementById('resultBox').style.display = 'none';
    document.getElementById('errorBox').style.display  = 'none';
    document.getElementById('progressWrap').style.display = 'none';
    setProgress(0, '');
  }
</script>
</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/crop', methods=['POST'])
def crop():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['pdf']
    if not f.filename.endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400

    file_id  = str(uuid.uuid4())
    in_path  = str(UPLOAD_DIR / f"{file_id}_input.pdf")
    out_name = Path(f.filename).stem + "_cropped.pdf"
    out_path = str(UPLOAD_DIR / f"{file_id}_output.pdf")

    f.save(in_path)

    try:
        crop_flipkart_labels(in_path, out_path)
    except Exception as e:
        os.remove(in_path)
        return jsonify({'error': str(e)}), 500

    os.remove(in_path)
    return jsonify({'file_id': file_id, 'filename': out_name})

@app.route('/download/<file_id>')
def download(file_id):
    # Sanitize
    file_id = file_id.replace('/', '').replace('..', '')
    out_path = UPLOAD_DIR / f"{file_id}_output.pdf"
    if not out_path.exists():
        return 'File not found', 404
    return send_file(str(out_path), as_attachment=True,
                     download_name=out_path.name.replace(f"{file_id}_", ""))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
