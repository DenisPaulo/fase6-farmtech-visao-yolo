"""Testes automatizados da Fase 6 (Visão Computacional YOLOv8 na lavoura)."""

from src.detector_lavoura import (
    CLASSES_YOLO,
    DETECCOES_JSON,
    PAINEL_PNG,
    executar_pipeline_yolo,
)


def test_executar_pipeline_yolo_detecta_pragas_e_gera_json():
    rel = executar_pipeline_yolo()
    assert DETECCOES_JSON.exists()
    assert PAINEL_PNG.exists()
    assert rel["imagens_analisadas"] == 4
    assert rel["total_objetos_detectados"] >= 8
    assert len(rel["alertas_fitossanitarios"]) >= 3
    classes_alerta = {a["classe_detectada"] for a in rel["alertas_fitossanitarios"]}
    assert {"ferrugem_cafeeiro", "praga_broca", "deficit_hidrico"} <= classes_alerta
    assert len(CLASSES_YOLO) == 5
