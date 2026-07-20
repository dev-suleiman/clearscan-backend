import io

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def test_image_bytes() -> bytes:
    img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
