"""AccentRoute 管线 CLI。各阶段命令随对应模块落地（T2–T14）陆续注册。"""

import typer

app = typer.Typer(name="accentroute", no_args_is_help=True)


@app.callback()
def main() -> None:
    """AccentRoute 数据管线。"""


@app.command()
def version() -> None:
    """打印版本。"""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("accentroute"))


if __name__ == "__main__":
    app()
