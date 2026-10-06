"""CLI da Guardiã AI (Typer) — reaproveita padrão do tech-4.

Comandos:
    train        Treina os 2+ modelos e imprime a comparação de métricas (#3/#4)
    predict      Analisa um caso e grava relatório + auditoria
    rag-ingest   Indexa protocolos/cartilhas de data/knowledge na base RAG (#7)
    report       Gera o relatório técnico em PDF (#13)
    demo         Sobe a interface Streamlit (#9)
"""

from __future__ import annotations

import json

import typer
from loguru import logger
from rich.console import Console

app = typer.Typer(add_completion=False, help="Guardiã AI — IA para saúde e segurança da mulher.")
console = Console()


@app.command()
def train(target: str = typer.Option(None, help="Nome da coluna alvo (padrão: settings)")):
    """Treina os modelos e mostra a comparação de métricas."""
    from src.ml.train import train_all

    from src.ml.evaluate import METRIC_RATIONALE

    table = train_all(target=target)
    console.print("\n[bold]Comparação de modelos (teste)[/bold]")
    console.print(table.round(3))
    console.print(f"\n{METRIC_RATIONALE}")
    console.print(f"\nModelo escolhido: [bold]{table.index[0]}[/bold]")
    logger.info("Treino concluído. Modelos salvos em models/.")


@app.command()
def rag_ingest(directory: str = typer.Argument("data/knowledge")):
    """Ingerir base de conhecimento no vector store (RAG)."""
    from src.rag.ingest import ingest_documents

    n = ingest_documents(directory)
    console.print(f"[green]{n} chunks indexados.[/green]")


@app.command()
def predict(
    features: str = typer.Option(..., help='JSON dos dados, ex.: {"age":28,"systolic_bp":138}'),
    report: str = typer.Option(None, help="Relato textual do atendimento"),
    model: str | None = typer.Option(
        None,
        help="Modelo treinado. Se omitido, usa o escolhido pelas métricas do treino.",
    ),
):
    """Rodar a jornada completa para um caso (ML -> RAG -> LLM) + salvar relatório."""
    from src.graph.workflow import analyze_case
    from src.services.audit_log import log_analysis
    from src.utils.report_generator import save_report

    result = analyze_case(json.loads(features), report_text=report, model_name=model)
    path = save_report(result)
    log_analysis(result)
    console.print_json(json.dumps({k: result[k] for k in ("prediction", "probability", "rag_sources")}))
    console.print(f"\n[bold]Explicação:[/bold]\n{result['explanation']}")
    console.print(f"\nRelatório salvo em: {path}")


@app.command()
def demo():
    """Subir a interface Streamlit."""
    import subprocess

    raise typer.Exit(
        subprocess.call(["streamlit", "run", "src/app.py", "--server.port", "8501"])
    )


@app.command("report")
def technical_report():
    """Gerar o relatório técnico da entrega em PDF."""
    from src.utils.technical_report import generate_technical_report

    path = generate_technical_report()
    console.print(f"[green]Relatório gerado em {path}.[/green]")


if __name__ == "__main__":
    app()
