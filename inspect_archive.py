from pathlib import Path
from zipfile import ZipFile

archive = Path(r"C:\Users\Devab\Downloads\archive.zip")
with ZipFile(archive) as zf:
    infos = zf.infolist()
    print("members", len(infos))
    for info in infos[:20]:
        print(info.filename)
    bad = [info.filename for info in infos if info.filename.startswith("/") or info.filename.startswith("\\") or ".." in Path(info.filename).parts]
    print("unsafe", len(bad))
    for name in bad[:20]:
        print("BAD", name)
