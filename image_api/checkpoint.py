import json
import shutil
from pathlib import Path
from threading import RLock
from uuid import uuid4

import folder_paths
from PIL import Image
from .. import generation_cache

_FILES_LOCK = RLock()


def write_json(path, value):
    with _FILES_LOCK:
        temporary = path.with_suffix('.json.part')
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)


class BatchCheckpoint:
    def __init__(self, batch_id, force_regenerate=False):
        session_id = generation_cache.CACHE.session_id
        root = Path(folder_paths.get_user_directory()) / 'llm_harness_image_resume' / session_id / batch_id
        root.mkdir(parents=True, exist_ok=True)
        active = root / 'active.json'
        with _FILES_LOCK:
            if force_regenerate or not active.exists():
                run_id = uuid4().hex
                write_json(active, {'run_id': run_id})
            else:
                run_id = json.loads(active.read_text(encoding='utf-8'))['run_id']
                if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
                    raise ValueError('续做记录损坏，请开启“重新生成”。原图片备份仍保留。')
        self.directory = root / run_id
        self.directory.mkdir(exist_ok=True)
        self.temp_subfolder = f'llm_harness_image_resume/{session_id}/{batch_id}/{run_id}'
        self.temp_directory = Path(folder_paths.get_temp_directory()) / self.temp_subfolder
        self.temp_directory.mkdir(parents=True, exist_ok=True)

    def image_path(self, index):
        return self.directory / f'{index + 1:04d}.png'

    def response_path(self, index):
        return self.directory / f'{index + 1:04d}.response.json'

    def load(self, index):
        with _FILES_LOCK:
            path = self.image_path(index)
            if not path.exists():
                return None
            try:
                with Image.open(path) as image:
                    result = image.convert('RGB')
            except (OSError, ValueError):
                # Keep the damaged PNG; the saved API response can reconstruct it.
                return None
            self.copy_to_temp(index)
        return result

    def save(self, index, image):
        with _FILES_LOCK:
            path = self.image_path(index)
            temporary = path.with_suffix('.png.part')
            image.save(temporary, format='PNG')
            temporary.replace(path)
            self.copy_to_temp(index)

    def copy_to_temp(self, index):
        with _FILES_LOCK:
            path = self.image_path(index)
            target = self.temp_directory / path.name
            temporary = target.with_suffix('.png.part')
            shutil.copyfile(path, temporary)
            temporary.replace(target)

    def progress(self, total, completed, active, failed, waiting, reused):
        status = {
            'total': total, 'completed': completed, 'active': active,
            'failed': failed, 'waiting': waiting, 'reused': reused,
            'backup_directory': str(self.directory),
            'temp_directory': str(self.temp_directory),
        }
        write_json(self.directory / 'progress.json', status)
        return status
