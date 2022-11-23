import unittest
import requests

from tests.test_data import UNBATCHED_INPUT_DATA, UNBATCHED_INPUT_LABEL, BATCHED_INPUT_DATA, BATCHED_LABELS


class TestClient(unittest.TestCase):
    def setUp(self) -> None:
        self.url = "http://172.17.0.1:8050/"

    def post_request(self, features, preprocess, endpoint):
        data = {
            "features": features,
            "query_size": preprocess,
        }
        return requests.post(self.url + endpoint, json=data)

    def test_server_ready(self):
        response = requests.get(self.url)
        self.assertEqual(response.status_code, 200)

    def test_infer(self):
        response = self.post_request(UNBATCHED_INPUT_DATA, preprocess=False, endpoint="infer")
        prediction = response.json().get("prediction")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(prediction), len(UNBATCHED_INPUT_LABEL))
        self.assertEqual(prediction, UNBATCHED_INPUT_LABEL)

    def test_batch_infer(self):
        response = self.post_request(BATCHED_INPUT_DATA, preprocess=False, endpoint="batch_infer")
        prediction = response.json().get("predictions")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(prediction), len(BATCHED_LABELS))
        self.assertEqual(prediction, BATCHED_LABELS)
