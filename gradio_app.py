"""
NeuroSim inference frontend (Gradio).

Run:
    uv run python gradio_app.py
    # or: uv run gradio gradio_app.py

Spawns `python -u inference.py --run_dir <per-run-dir> ...` as a subprocess,
streams its stdout to the UI, and bundles the run directory + layer_record
+ NetWork CSV into a downloadable zip.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import signal
import subprocess
import sys
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE = PROJECT_ROOT / "inference.py"
RUNS_ROOT = PROJECT_ROOT / "gradio_runs"
RUNS_ROOT.mkdir(exist_ok=True)

# At most one concurrent run; protected by a lock.
_run_lock = threading.Lock()
_proc_holder: dict[str, subprocess.Popen | None] = {"proc": None}

# Linux: tell the kernel to SIGTERM the child the instant this process dies,
# even on SIGKILL — the only safety net that survives `kill -9` on the parent.
_PR_SET_PDEATHSIG = 1
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None


def _child_preexec() -> None:
    if _libc is not None:
        _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


def _resolve_upload(uploaded, fallback_path: str, target_name: str, run_dir: Path) -> str:
    """If a file was uploaded, copy it into run_dir and return that path; else return fallback."""
    if uploaded is None:
        return fallback_path
    src = Path(uploaded if isinstance(uploaded, str) else uploaded.name)
    dst = run_dir / target_name
    shutil.copyfile(src, dst)
    return str(dst)


def _build_cmd(run_dir: Path, values: dict) -> list[str]:
    cmd = [sys.executable, "-u", str(INFERENCE), "--run_dir", str(run_dir)]
    for k, v in values.items():
        if v is None:
            continue
        cmd.extend([f"--{k}", str(v)])
    return cmd


def _zip_results(run_dir: Path, model: str) -> Path:
    """Bundle run_dir + layer_record_<model> + NetWork_<model>.csv."""
    zip_path = run_dir / "results.zip"
    record_dir = PROJECT_ROOT / f"layer_record_{model}"
    network_csv = PROJECT_ROOT / "NeuroSIM" / f"NetWork_{model}.csv"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                zf.write(p, arcname=str(p.relative_to(run_dir)))
        if record_dir.exists():
            for p in record_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=f"layer_record_{model}/{p.relative_to(record_dir)}")
        if network_csv.exists():
            zf.write(network_csv, arcname=f"NeuroSIM/NetWork_{model}.csv")
    return zip_path


def run_inference(
    # Core
    dataset, model, data_path, model_path, test_name,
    gpu, batch_size, calib_batch_size, seed,
    # Hardware
    hardware, ppa, num_batches,
    sub_array_r, sub_array_c, parallel_read,
    bitcell, mem_type, off_state, on_state,
    mem_states_file_path, mem_states_upload, vdd,
    # Noise / reliability
    read_noise, output_noise,
    output_noise_file_path, output_noise_upload,
    t, v, detect, target, rate_stuck_0, rate_stuck_1,
    # Quant
    weight_precision, input_precision, dac_precision, adc_precision,
    input_calib_method, weight_calib_method, adc_calib_method,
    adc_quant_method, adc_enable, optimize_adc,
):
    if not _run_lock.acquire(blocking=False):
        yield "이미 실행 중인 작업이 있습니다.", None, None
        return

    try:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True)

        mem_states = _resolve_upload(mem_states_upload, mem_states_file_path or "", "mem_states.csv", run_dir)
        noise_file = _resolve_upload(output_noise_upload, output_noise_file_path or "", "output_noise.csv", run_dir)

        values = {
            "dataset": dataset,
            "model": model,
            "data_path": data_path,
            "model_path": model_path,
            "test_name": test_name,
            "gpu": int(gpu),
            "batch_size": int(batch_size),
            "calib_batch_size": int(calib_batch_size),
            "seed": int(seed),
            "hardware": int(hardware),
            "ppa": int(ppa),
            "num_batches": int(num_batches),
            "sub_array": f"[{int(sub_array_r)},{int(sub_array_c)}]",
            "parallel_read": int(parallel_read),
            "bitcell": int(bitcell),
            "mem_type": mem_type,
            "off_state": float(off_state),
            "on_state": float(on_state),
            "mem_states_file": mem_states,
            "vdd": float(vdd),
            "read_noise": float(read_noise),
            "output_noise": float(output_noise),
            "output_noise_file": noise_file,
            "t": float(t),
            "v": float(v),
            "detect": int(detect),
            "target": float(target),
            "rate_stuck_0": float(rate_stuck_0),
            "rate_stuck_1": float(rate_stuck_1),
            "weight_precision": int(weight_precision),
            "input_precision": int(input_precision),
            "dac_precision": int(dac_precision),
            "adc_precision": int(adc_precision),
            "input_calib_method": input_calib_method,
            "weight_calib_method": weight_calib_method,
            "adc_calib_method": adc_calib_method,
            "adc_quant_method": adc_quant_method,
            "adc_enable": int(adc_enable),
            "optimize_adc": int(optimize_adc),
        }

        cmd = _build_cmd(run_dir, values)
        log_path = run_dir / "stdout.log"

        header = f"[run_dir] {run_dir}\n[command]\n{' '.join(cmd)}\n\n"
        accumulated = [header]
        log_path.write_text(header, encoding="utf-8")
        yield "".join(accumulated), str(log_path), None

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            preexec_fn=_child_preexec,
            start_new_session=True,
        )
        _proc_holder["proc"] = proc

        with log_path.open("a", encoding="utf-8") as log_f:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                accumulated.append(line)
                log_f.write(line)
                log_f.flush()
                yield "".join(accumulated), str(log_path), None

        proc.wait()
        _proc_holder["proc"] = None

        footer = f"\n[exit code] {proc.returncode}\n"
        accumulated.append(footer)
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(footer)

        zip_path = _zip_results(run_dir, str(model))
        yield "".join(accumulated), str(log_path), str(zip_path)
    finally:
        _proc_holder["proc"] = None
        _run_lock.release()


def _killpg_safely(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass


def stop_inference():
    proc = _proc_holder.get("proc")
    if proc is None or proc.poll() is not None:
        return "실행 중인 작업이 없습니다."
    try:
        _killpg_safely(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _killpg_safely(proc, signal.SIGKILL)
            proc.wait(timeout=3)
        return f"중지됨 (exit code {proc.returncode})."
    except Exception as e:
        return f"중지 실패: {e}"


@atexit.register
def _kill_lingering_proc() -> None:
    proc = _proc_holder.get("proc")
    if proc is None or proc.poll() is not None:
        return
    _killpg_safely(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _killpg_safely(proc, signal.SIGKILL)


def build_ui():
    _autoscroll_js = """
function() {
    function hookLogScroll() {
        const el = document.querySelector('#log_box textarea');
        if (!el) { setTimeout(hookLogScroll, 500); return; }
        const proto = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
        Object.defineProperty(el, 'value', {
            set(v) { proto.set.call(this, v); this.scrollTop = this.scrollHeight; },
            get() { return proto.get.call(this); }
        });
    }
    hookLogScroll();
}
"""
    with gr.Blocks(title="NeuroSim", js=_autoscroll_js) as demo:
        gr.Markdown("# NeuroSim Inference")
        gr.Markdown("`inference.py` 를 서브프로세스로 호출합니다. `--run_dir` 패치 필요.")

        with gr.Tabs():
            with gr.Tab("Core"):
                dataset = gr.Dropdown(["cifar10", "cifar100", "imagenet", "mnist"], value="cifar10", label="dataset",
                                      info="cifar10|cifar100|imagenet|mnist")
                model = gr.Dropdown(["vgg8", "resnet18", "resnet50", "resnet101", "resnet152", "swin_t", "lenet"], value="vgg8", label="model",
                                    info="vgg8|resnet18|resnet50|swin_t|lenet (mnist에만 호환)")
                data_path = gr.Textbox("./datasets/", label="data_path",
                                       info="path to saved datasets")
                model_path = gr.Textbox("./models/", label="model_path",
                                        info="path to saved models")
                test_name = gr.Textbox("test", label="test_name",
                                       info="test name")
                gpu = gr.Number(0, label="gpu", precision=0,
                                info="GPU id to use. Only single GPU support for now")
                batch_size = gr.Number(400, label="batch_size", precision=0,
                                       info="input batch size for inference (default: 64)")
                calib_batch_size = gr.Number(128, label="calib_batch_size", precision=0,
                                             info="input batch size for calibrating quantization (default: 128)")
                seed = gr.Number(1234, label="seed", precision=0,
                                 info="random seed (default: 1)")

            with gr.Tab("Hardware"):
                hardware = gr.Radio([0, 1], value=1, label="hardware (0=off, 1=on)",
                                    info="run hardware inference simulation")
                ppa = gr.Radio([0, 1], value=1, label="ppa (run C++ analysis)",
                               info="run power, performance, and area analysis (C++)")
                num_batches = gr.Number(-1, label="num_batches (-1 = full dataset)", precision=0,
                                        info="number of batches to run in hardware inference simulation, -1 for full dataset")
                with gr.Row():
                    sub_array_r = gr.Number(128, label="sub_array rows", precision=0,
                                            info="size of subArray (e.g. 128x128) — rows")
                    sub_array_c = gr.Number(128, label="sub_array cols", precision=0,
                                            info="size of subArray (e.g. 128x128) — cols")
                parallel_read = gr.Number(128, label="parallel_read", precision=0,
                                          info="number of rows read in parallel (<= subArray e.g. 32)")
                bitcell = gr.Number(1, label="bitcell", precision=0,
                                    info="cell precision (e.g. 4-bit/cell)")
                mem_type = gr.Dropdown(["resistive", "capacitive"], value="resistive", label="mem_type",
                                       info="memory cell type: resistive | capacitive")
                off_state = gr.Textbox("1e-15", label="off_state (S 또는 F)",
                                       info="device off state (conductance (S) or capacitance (F))")
                on_state = gr.Textbox("1e-14", label="on_state (S 또는 F)",
                                      info="device on conductance / capacitance")
                mem_states_file_path = gr.Textbox("mem_states.csv", label="mem_states_file 경로 (빈 문자열 = on/off만 사용)",
                                                  info="path to .csv file containing mean and std for each memory state")
                mem_states_upload = gr.File(label="mem_states.csv 업로드 (선택, 우선순위 높음)", file_types=[".csv"])
                vdd = gr.Number(1.0, label="vdd",
                                info="supply voltage")

            with gr.Tab("Noise / Reliability"):
                read_noise = gr.Number(0.0, label="read_noise",
                                       info="std of conductance read noise, derive from SPICE or experiment, or use to test reliability")
                output_noise = gr.Number(0.0, label="output_noise (-1 = 파일에서 std 로드)",
                                         info="std of output voltage noise, derive from SPICE or experiment, or use to test reliability")
                output_noise_file_path = gr.Textbox("", label="output_noise_file 경로",
                                                    info="path and file to saved output std")
                output_noise_upload = gr.File(label="output_noise.csv 업로드 (선택)", file_types=[".csv"])
                t = gr.Number(1.0, label="t (retention time)",
                              info="retention time")
                v = gr.Number(0.0, label="v (drift coefficient)",
                              info="drift coefficient")
                detect = gr.Radio([0, 1], value=0, label="detect (0=random, 1=fixed)",
                                  info="if 1, fixed-direction drift, if 0, random drift")
                target = gr.Number(0.0, label="target",
                                   info="drift target for fixed-direction drift, range 0-1")
                rate_stuck_0 = gr.Number(0.0, label="rate_stuck_0",
                                         info="rate of cells stuck at 0, range 0-1, default 0")
                rate_stuck_1 = gr.Number(0.0, label="rate_stuck_1",
                                         info="rate of cells stuck at 1, range 0-1, default 0")

            with gr.Tab("Quantization"):
                weight_precision = gr.Number(8, label="weight_precision", precision=0,
                                             info="number of bits to quantize the weights to (integer)")
                input_precision = gr.Number(8, label="input_precision", precision=0,
                                            info="number of bits to quantize the inputs to (integer)")
                dac_precision = gr.Number(1, label="dac_precision", precision=0,
                                          info="DAC precision (e.g. 1-bit)")
                adc_precision = gr.Number(7, label="adc_precision", precision=0,
                                          info="ADC precision (e.g. 7-bit)")
                input_calib_method = gr.Dropdown(["max", "histogram"], value="max", label="input_calib_method",
                                                 info="histogram or max")
                weight_calib_method = gr.Dropdown(["max", "histogram"], value="max", label="weight_calib_method",
                                                  info="histogram or max")
                adc_calib_method = gr.Dropdown(["max", "histogram"], value="max", label="adc_calib_method",
                                               info="histogram or max")
                adc_quant_method = gr.Textbox("scale", label="adc_quant_method",
                                              info="ADC quantization scaling method (default: scale)")
                adc_enable = gr.Radio([0, 1], value=1, label="adc_enable",
                                      info="enable ADC quantization (0=disabled, 1=enabled)")
                optimize_adc = gr.Radio([0, 1], value=0, label="optimize_adc",
                                        info="optimize ADC quantization (0=off, 1=on)")

        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            stop_btn = gr.Button("Stop")

        log_box = gr.Textbox(label="실시간 로그", lines=25, max_lines=25, autoscroll=True, elem_id="log_box")
        copy_btn = gr.Button("로그 복사", size="sm")

        with gr.Row():
            log_dl = gr.File(label="stdout.log 다운로드", interactive=False)
            result_dl = gr.File(label="결과 zip 다운로드", interactive=False)

        status_box = gr.Textbox(label="상태", interactive=False)

        run_inputs = [
            dataset, model, data_path, model_path, test_name,
            gpu, batch_size, calib_batch_size, seed,
            hardware, ppa, num_batches,
            sub_array_r, sub_array_c, parallel_read,
            bitcell, mem_type, off_state, on_state,
            mem_states_file_path, mem_states_upload, vdd,
            read_noise, output_noise,
            output_noise_file_path, output_noise_upload,
            t, v, detect, target, rate_stuck_0, rate_stuck_1,
            weight_precision, input_precision, dac_precision, adc_precision,
            input_calib_method, weight_calib_method, adc_calib_method,
            adc_quant_method, adc_enable, optimize_adc,
        ]
        run_btn.click(run_inference, inputs=run_inputs, outputs=[log_box, log_dl, result_dl])
        stop_btn.click(stop_inference, outputs=status_box)
        copy_btn.click(
            fn=None,
            inputs=log_box,
            outputs=None,
            js="async (v) => { try { await navigator.clipboard.writeText(v ?? ''); } catch (e) { console.error(e); } return []; }",
        )

    return demo


demo = build_ui().queue()


def _parse_accounts() -> list[tuple[str, str]] | None:
    raw = os.environ.get("APP_ACCOUNTS", "")
    accounts = []
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        user, _, pw = pair.partition(":")
        if user and pw:
            accounts.append((user, pw))
    return accounts or None


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True,
                auth=_parse_accounts(), auth_message="NeuroSim — 로그인이 필요합니다.")
