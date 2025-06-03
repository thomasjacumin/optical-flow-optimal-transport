<!-- # optical-flow-optimal-transport

python -m venv ~/foto
source ~/foto/bin/activate

pip install -r requirements.txt -->

---

# Optical Flow Benchmark Pipeline

This repository provides an automated pipeline for evaluating optical flow algorithms on the [Middlebury benchmark dataset](https://vision.middlebury.edu/flow/). It supports two optical flow methods:
- **Gennert and Negahdaripour (GN)**
- **FOTO**

The pipeline downloads the datasets, runs evaluations, and visualizes the results using color-coded flow maps.

---

## 📦 Features

- Automatic download and unpacking of Middlebury datasets.
- Execution of optical flow algorithms (`GN` and `FOTO`) on each dataset.
- Output of `.flo` files, reconstruction images, benchmark statistics, and visualizations.
- Ground truth-based normalization on Middlebury-2 data.

---

### 📂 Directory Structure

After running the pipeline, the following structure will be created:

```
.
├── data/
│   ├── middlebury-1/
│   │   └── eval-data-gray/
│   │       └── <sequence>/
│   │           ├── frame10.png
│   │           └── frame11.png
│   └── middlebury-2/
│       ├── other-data-gray/
│       │   └── <sequence>/
│       │       ├── frame10.png
│       │       └── frame11.png
│       └── other-gt-flow/
│           └── <sequence>/
│               └── flow10.flo
│
├── results/
│   ├── middlebury-1/
│   │   └── <sequence>/
│   │       ├── gn.flo
│   │       ├── gn.png
│   │       ├── gn.rec.png
│   │       ├── gn.benchmark.txt
│   │       ├── foto.flo
│   │       ├── foto.png
│   │       ├── foto.rec.png
│   │       └── foto.benchmark.txt
│   └── middlebury-2/
│       └── <sequence>/
│           ├── flow10.png        # Ground truth visualization
│           ├── gn.flo
│           ├── gn.png
│           ├── gn.rec.png
│           ├── gn.benchmark.txt
│           ├── foto.flo
│           ├── foto.png
│           ├── foto.rec.png
│           └── foto.benchmark.txt
```

* `<sequence>` stands for the name of each image pair directory, e.g., `Grove2`, `Urban2`, etc.
- `*.flo` (flow output)
- `*.benchmark.txt` (evaluation stats)
- `*.rec.png` (reconstruction)
- `*.lum.png` (luminance)
- `*.png` (color-encoded flow visualization)

---

## 🚀 Usage

First, make the script executable:

```bash
chmod +x run.sh
````

Then, use the following commands:

### Download Middlebury Datasets

```bash
./run.sh download
```

### Install Python Dependencies

Ensure you have Python 3 and `pip`, then:

```bash
./run.sh install
```

### Run the Pipeline

To process all datasets and compute optical flow:

```bash
./run.sh
```

### Restart (Clean Previous Results)

To delete previous results and rerun everything:

```bash
./run.sh restart
```

### Display Help

```bash
./run.sh help
```

---

## 🧪 Dependencies

Python packages (install via `requirements.txt`):

* `numpy`
* `Pillow`
* `matplotlib`
* `scipy`

Other tools:

* `wget`, `unzip` (for dataset handling)
* `bash`, `python3`

Ensure `./bin/color_flow` is present and executable.

---

## ⚠️ Notes

* The script uses flags (e.g., `.out.gn.sucess`) to avoid recomputing results. Remove these manually to re-run specific tests.
* Ground truth normalization is computed and applied only when ground truth `.flo` files exist.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.
You are free to use, modify, and distribute this software under the terms of the GPLv3.
See the [LICENSE](./LICENSE) file for the full license text.

---

## 🙏 Acknowledgments

* [Middlebury Optical Flow Evaluation](https://vision.middlebury.edu/flow/)
