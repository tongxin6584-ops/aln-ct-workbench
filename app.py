"""云端界面：提交后台任务，重连查看进度，浏览历史报告。"""

import json
import re
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import streamlit as st

from modules.jobs import list_jobs, list_results, start_job
from modules.report import render_markdown

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ALN Product", layout="wide")
st.title("腋窝淋巴结 CT 分析")
st.caption("上传全画幅静脉期胸部 CT NIfTI；云端执行分析。病例编号请先脱敏。")


def _tail(path, count=70):
    path = Path(path)
    if not path.is_file():
        return "等待日志生成…"
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=count))


def _show_job():
    jobs = list_jobs(ROOT)
    if not jobs:
        st.info("暂无网页提交的任务；下方仍可打开命令行运行的历史结果。")
        return
    labels = [f"{job['patient_id']} / {job['run_id']} / {job['status']}" for job in jobs]
    selected = st.selectbox("选择任务", range(len(jobs)), format_func=lambda i: labels[i], key="job_index")
    job = jobs[selected]
    if job["status"] == "running":
        st.info("分析正在服务器后台运行。关闭网页或断开 SSH 后，可再次打开本页查看。")
    elif job["status"] == "completed":
        st.success(f"分析完成：{job['result_dir']}")
    else:
        st.error(f"任务状态：{job['status']}；请查看下方日志。")
    st.code(_tail(job["console_log"]), language="text")


def _show_result():
    reports = list_results(ROOT)
    if not reports:
        st.info("尚无已完成的报告。")
        return
    labels = [f"{path.parent.parent.name} / {path.parent.name}" for path in reports]
    selected = st.selectbox("选择历史结果", range(len(reports)), format_func=lambda i: labels[i], key="result_index")
    report_path = reports[selected]
    result = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    nodes = report.get("nodes", [])
    st.subheader(f"结果：{len(nodes)} 颗淋巴结")
    st.caption(f"结果目录：{result}")
    st.dataframe(nodes, use_container_width=True, hide_index=True)
    if nodes:
        by_id = {node["node_id"]: node for node in nodes}
        node_id = st.selectbox("查看单颗淋巴结", list(by_id))
        st.json(by_id[node_id])
    md_path = report_path.with_suffix(".md")
    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else render_markdown(report)
    if not md_path.exists():
        st.caption("这是旧版运行记录。Markdown 在网页中临时生成，原报告缺少的版本和时间字段显示为“未记录”。")
    with st.expander("阅读结构化报告", expanded=False):
        st.markdown(markdown)
    st.download_button("下载 Markdown 报告", markdown.encode("utf-8"),
                       file_name=md_path.name, mime="text/markdown", key=f"md_{report_path}")
    overview = result / "previews" / "overview.png"
    montage = result / "previews" / "node_montage.png"
    if overview.exists():
        st.image(str(overview), caption="原始全画幅 CT 叠加实例标签")
    if montage.exists():
        st.image(str(montage), caption=("历史拼图可能只含前12颗；上方表格含全部节点"
                                        if not md_path.exists() else "全部单颗淋巴结预览"))
    st.markdown("#### 下载结果文件")
    for path in sorted(result.iterdir()):
        if path.is_file() and path != md_path:
            with path.open("rb") as handle:
                st.download_button(f"下载 {path.name}", handle, file_name=path.name,
                                   key=f"download_{path}")


tab_submit, tab_jobs, tab_results = st.tabs(["新病例", "运行进度", "历史结果"])
with tab_submit:
    with st.form("case_form"):
        upload = st.file_uploader("CT 文件（.nii 或 .nii.gz）", type=["nii", "gz"])
        patient_id = st.text_input("脱敏病例编号", value="patient_001")
        side = st.selectbox("目标侧别", ["left", "right", "bilateral"])
        submit = st.form_submit_button("开始分析")
    if submit:
        if upload is None:
            st.error("请先上传 NIfTI 文件")
        elif not re.fullmatch(r"[A-Za-z0-9_-]+", patient_id):
            st.error("编号只能含英文字母、数字、下划线和连字符")
        elif not upload.name.endswith((".nii", ".nii.gz")):
            st.error("文件必须以 .nii 或 .nii.gz 结尾")
        else:
            run_id = datetime.now().strftime("web_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            upload_dir = ROOT / "workspace" / "_uploads" / run_id
            upload_dir.mkdir(parents=True)
            input_path = upload_dir / ("input.nii.gz" if upload.name.endswith(".nii.gz") else "input.nii")
            with input_path.open("wb") as output:
                while chunk := upload.read(4 * 1024 * 1024):
                    output.write(chunk)
            try:
                start_job(ROOT, input_path, patient_id, side, run_id)
            except Exception as exc:
                st.error(f"任务提交失败：{type(exc).__name__}: {exc}")
            else:
                st.success(f"已提交后台任务：{run_id}。请切换到“运行进度”；断线后重新打开本页即可查看。")
with tab_jobs:
    if hasattr(st, "fragment"):
        @st.fragment(run_every="5s")
        def refresh_jobs():
            _show_job()
        refresh_jobs()
    else:
        st.caption("当前 Streamlit 版本不支持自动刷新；点击下方按钮更新。")
        if st.button("刷新运行进度"):
            st.rerun()
        _show_job()
with tab_results:
    _show_result()
