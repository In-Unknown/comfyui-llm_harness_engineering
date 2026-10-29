"""Standalone PySide6 dialog for managing LLM connections."""
import json
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


def _get_cfg_path():
    here = Path(__file__).resolve().parent
    proj = here.parent
    data_dir = proj / "data"
    return data_dir, data_dir / "connections.json"


def _load(path):
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(value, dict) for value in data.values()):
        raise ValueError("连接配置必须是按连接名保存的 JSON 对象。")
    return data


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _update(path, name, config):
    data = _load(path)
    if config is None:
        data.pop(name, None)
    else:
        data[name] = config
    _save(path, data)
    return data


def main():
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QFormLayout,
        QLineEdit, QComboBox, QDialogButtonBox, QLabel, QMessageBox,
        QListWidget,
    )
    from PySide6.QtCore import Qt

    data_dir, cfg_file = _get_cfg_path()
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        data = _load(cfg_file)
    except (OSError, ValueError) as error:
        QMessageBox.critical(None, "读取配置失败", f"原文件未修改。请检查 {cfg_file}：\n{error}")
        return

    dlg = QDialog()
    dlg.setWindowTitle("LLM Harness - API 来源 / 连接管理")
    dlg.setMinimumWidth(680)
    dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)

    root = QVBoxLayout(dlg)
    root.addWidget(QLabel("已有连接（点击加载，或新建）:"))
    lst = QListWidget()
    for n in sorted(data.keys()):
        lst.addItem(n)
    lst.setMaximumHeight(120)
    root.addWidget(lst)

    form = QFormLayout()
    name_e = QLineEdit()
    name_e.setPlaceholderText("例如: deepseek / anthropic / local")
    form.addRow("连接名（英数字/_-）:", name_e)
    fmt_c = QComboBox()
    fmt_c.addItems(["Chat Completions", "Responses", "Anthropic Messages"])
    fmt_c.setToolTip("此项用于 LLM 远程回复；API图片编辑和API图像生成使用图像接口，不读取此选项。")
    form.addRow("API 格式（文本节点）:", fmt_c)
    ep_e = QLineEdit()
    ep_e.setPlaceholderText("例如 https://api.openai.com/v1  或 http://127.0.0.1:11434/v1")
    form.addRow("Endpoint:", ep_e)
    key_e = QLineEdit()
    key_e.setEchoMode(QLineEdit.Password)
    key_e.setPlaceholderText("sk-... （本地服务可留空）")
    form.addRow("API Key:", key_e)
    model_e = QLineEdit()
    model_e.setPlaceholderText("例如 gpt-6-sol, gpt-image-2 （多个用英文逗号分隔）")
    form.addRow("模型 ID:", model_e)
    root.addLayout(form)

    def load_selected(item):
        if item is None:
            return
        name = item.text()
        cfg = data.get(name, {})
        name_e.setText(name)
        idx = fmt_c.findText(cfg.get("api_format", "Chat Completions"))
        fmt_c.setCurrentIndex(max(0, idx))
        ep_e.setText(cfg.get("endpoint", ""))
        key_e.setText(cfg.get("api_key", ""))

        models = cfg.get("models")
        if not models and cfg.get("model"):
            models = [cfg["model"]]
        if models:
            model_e.setText(", ".join(models))
        else:
            model_e.setText("")

    lst.itemClicked.connect(load_selected)

    def refresh_list():
        lst.clear()
        for n in sorted(data.keys()):
            lst.addItem(n)

    def do_save():
        nonlocal data
        name = name_e.text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", name):
            QMessageBox.warning(dlg, "错误", "连接名必须为 1-64 位英数字 / _ / -")
            return
        ep = ep_e.text().strip()
        if not ep:
            QMessageBox.warning(dlg, "错误", "Endpoint 不能为空")
            return

        mdl_str = model_e.text().strip()
        models = [m.strip() for m in mdl_str.split(",") if m.strip()]
        if not models:
            QMessageBox.warning(dlg, "错误", "模型 ID 不能为空")
            return

        config = {
            "api_format": fmt_c.currentText(),
            "endpoint": ep,
            "api_key": key_e.text(),
            "models": models,
        }
        try:
            data = _update(cfg_file, name, config)
        except (OSError, ValueError) as error:
            QMessageBox.warning(dlg, "保存失败", str(error))
            return
        refresh_list()
        QMessageBox.information(dlg, "完成", f"已保存来源 '{name}'。\n回到 ComfyUI 后自动刷新，也可点击节点上的刷新按钮。")
        dlg.accept()

    def do_delete():
        nonlocal data
        item = lst.currentItem()
        if not item:
            QMessageBox.warning(dlg, "提示", "请先在列表中选择要删除的连接")
            return
        name = item.text()
        if QMessageBox.question(dlg, "确认", f"删除连接 '{name}'") != QMessageBox.Yes:
            return
        try:
            data = _update(cfg_file, name, None)
        except (OSError, ValueError) as error:
            QMessageBox.warning(dlg, "删除失败", str(error))
            return
        refresh_list()
        QMessageBox.information(dlg, "完成", "已删除")

    btns = QDialogButtonBox()
    save_btn = btns.addButton("保存", QDialogButtonBox.AcceptRole)
    del_btn = btns.addButton("删除选中", QDialogButtonBox.DestructiveRole)
    close_btn = btns.addButton("关闭", QDialogButtonBox.RejectRole)
    save_btn.clicked.connect(do_save)
    del_btn.clicked.connect(do_delete)
    close_btn.clicked.connect(dlg.reject)
    root.addWidget(btns)

    dlg.exec()


if __name__ == "__main__":
    main()
