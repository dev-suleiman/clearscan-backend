import pytest


@pytest.mark.asyncio
async def test_compare_returns_winner(client, test_image_bytes):
    response = await client.post(
        "/api/v1/compare",
        files={"file": ("xray.png", test_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["winner"] in ["clahe", "cnn"]
