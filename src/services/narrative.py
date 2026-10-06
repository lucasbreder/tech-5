"""Organização do relato escrito para o profissional.

Lista transparente de termos. Não conclui que houve violência e não
aciona ninguém: só destaca trechos que merecem escuta com atenção.
"""

from __future__ import annotations

_TERMS: list[tuple[str, str]] = [
    ("ameaç", "menção a ameaça"),
    ("agress", "menção a agressão"),
    ("me bate", "menção a violência física"),
    ("bateu", "menção a violência física"),
    ("soco", "menção a violência física"),
    ("empurr", "menção a violência física"),
    ("medo", "expressão de medo"),
    ("não posso sair", "relato de restrição"),
    ("nao posso sair", "relato de restrição"),
    ("me proib", "relato de restrição"),
    ("controla", "relato de controle"),
    ("ciúme", "relato de ciúme associado a controle"),
    ("ciume", "relato de ciúme associado a controle"),
    ("arma", "menção a arma"),
    ("estupro", "menção a violência sexual"),
    ("forçou", "menção a coerção"),
    ("forcou", "menção a coerção"),
    ("ninguém ligar", "pedido para não contatar terceiros"),
    ("ninguem ligar", "pedido para não contatar terceiros"),
    ("não conta", "pedido de sigilo"),
    ("nao conta", "pedido de sigilo"),
]


def scan_report(text: str | None) -> dict:
    """Devolve sinais encontrados no relato, sem juízo sobre o fato."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "present": False,
            "signals": [],
            "note": "Nenhum relato foi informado neste atendimento.",
        }

    lowered = cleaned.lower()
    found: list[str] = []
    for stem, label in _TERMS:
        if stem in lowered and label not in found:
            found.append(label)

    if found:
        note = (
            "O relato traz termos da lista de atenção. Isso organiza a escuta; "
            "não confirma o fato nem autoriza decisão automática."
        )
    else:
        note = (
            "Nenhum termo da lista de atenção apareceu. A leitura do relato "
            "continua inteiramente com o profissional."
        )
    return {"present": True, "signals": found, "note": note}
