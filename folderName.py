from pathlib import Path
import pathspec


def load_gitignore(root: Path):
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None

    with gitignore.open("r", encoding="utf-8") as f:
        patterns = f.read().splitlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def folder_to_markdown(root_dir, output_md="structure.md"):
    spec = load_gitignore(root_dir)

    lines = [f"# Directory structure of `{root_dir.name}`\n"]

    def is_ignored(path: Path):
        if spec is None:
            return False
        rel = path.relative_to(root_dir)
        return spec.match_file(rel.as_posix())

    def walk(path, depth=0):
        indent = "  " * depth
        for item in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if is_ignored(item):
                continue

            if item.is_dir():
                lines.append(f"{indent}- 📁 **{item.name}/**")
                walk(item, depth + 1)
            else:
                lines.append(f"{indent}- 📄 {item.name}")

    walk(root_dir)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    folder_to_markdown(root, "output.md")
