import numpy as np
import cv2
import pytest


@pytest.mark.asyncio
async def test_assess_valid_image(client, test_image_bytes):
    response = await client.post(
        "/api/v1/assess",
        files={"file": ("xray.png", test_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quality_class"] in ["Good", "Fair", "Poor"]


@pytest.mark.asyncio
async def test_assess_invalid_extension(client, test_image_bytes):
    response = await client.post(
        "/api/v1/assess",
        files={"file": ("malware.exe", test_image_bytes, "application/octet-stream")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_assess_oversized_file(client):
    large_data = b"\x00" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    response = await client.post(
        "/api/v1/assess",
        files={"file": ("big.png", large_data, "image/png")},
    )
    assert response.status_code == 413
