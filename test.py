import unittest

from app import app


class FlaskTestCase(unittest.TestCase):

    # Ensure that Flask was set up correctly
    def test_index(self):
        tester=app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)
    # Ensure that responses are correct
    def test_index_post(self):
        tester=app.test_client(self)
        response = tester.post('/',
            data=dict(SERVER_ID="123", CPU_UTILIZATION="0", MEMORY_UTILIZATION="0", DISK_UTILIZATION="0"),
            follow_redirects=True)
        self.assertIn(b'No Alert,123', response.data)
        response = tester.post('/',
            data=dict(SERVER_ID="123", CPU_UTILIZATION="100", MEMORY_UTILIZATION="100", DISK_UTILIZATION="100"),
            follow_redirects=True)
        self.assertIn(b'Alert,123,CPU UTILIZATION VIOLATED,MEMORY UTILIZATION VIOLATED,DISK UTILIZATION VIOLATED', response.data)




if __name__ == '__main__':
    unittest.main()
