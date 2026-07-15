import unittest
import json
import os
import pandas as pd
from app import app

class OptiCropTestCase(unittest.TestCase):

    def setUp(self):
        # Set Flask to testing mode
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"OptiCrop", response.data)
        self.assertIn(b"Nitrogen", response.data)

    def test_prediction_endpoint(self):
        # Post realistic parameters
        post_data = {
            "N": "80",
            "P": "45",
            "K": "40",
            "temperature": "24.5",
            "humidity": "82.0",
            "ph": "6.5",
            "rainfall": "220.0",
            "location": "Coastal Delta",
            "season": "Kharif"
        }
        response = self.client.post('/predict', data=post_data)
        self.assertEqual(response.status_code, 200)
        # Verify result content is outputted
        self.assertIn(b"Recommended Best Crop", response.data)
        self.assertIn(b"Yield", response.data)
        self.assertIn(b"Soil Classification", response.data)

        # Check if saved to user_predictions.csv
        history_file = "datasets/user_predictions.csv"
        self.assertTrue(os.path.exists(history_file))
        df = pd.read_csv(history_file)
        self.assertGreater(len(df), 0)

    def test_dashboard_page(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Performance Analytics", response.data)
        self.assertIn(b"Accuracy", response.data)

    def test_health_api(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "online")
        self.assertIn("clf", data["models_loaded"])

if __name__ == '__main__':
    unittest.main()
