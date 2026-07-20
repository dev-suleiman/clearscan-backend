import pytest


@pytest.mark.asyncio
async def test_enhance_clahe_returns_base64(client, test_image_bytes):
    response = await client.post(
        "/api/v1/enhance/clahe",
        files={"file": ("xray.png", test_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "enhanced_image_b64" in data
    assert len(data["enhanced_image_b64"]) > 0
