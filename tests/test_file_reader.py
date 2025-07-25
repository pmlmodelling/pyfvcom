import os

import numpy.testing as test
import numpy as np

from unittest import TestCase
from datetime import datetime
from dateutil.relativedelta import relativedelta

from PyFVCOM.read import FileReader

from helpers.utils import _prep, StubFile


class FileReader_test(TestCase):

    def setUp(self):
        self.starttime, self.endtime, self.interval, self.lon, self.lat, self.triangles = _prep()
        self.stub = StubFile(self.starttime, self.endtime, self.interval,
                             lon=self.lon, lat=self.lat, triangles=self.triangles, zone='30N')
        self.reference = FileReader(self.stub.ncfile.name, variables=['ww', 'zeta', 'temp', 'h'])

    def tearDown(self):
        self.stub.ncfile.close()
        os.remove(self.stub.ncfile.name)
        del(self.stub)

    def test_get_single_lon(self):
        result = FileReader(self.stub.ncfile.name, dims={'node': [0]})
        test.assert_almost_equal(result.grid.lon, self.reference.grid.lon[0], decimal=5)

    def test_get_single_lat(self):
        result = FileReader(self.stub.ncfile.name, dims={'node': [29]})
        test.assert_almost_equal(result.grid.lat, self.reference.grid.lat[29], decimal=5)

    def test_get_single_lonc(self):
        result = FileReader(self.stub.ncfile.name, dims={'nele': [0]})
        test.assert_almost_equal(result.grid.lonc, self.reference.grid.lonc[0], decimal=5)

    def test_get_single_latc(self):
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29]})
        test.assert_almost_equal(F.grid.latc, self.reference.grid.latc[29], decimal=5)

    def test_get_multipe_lon(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [0, 5]})
        test.assert_almost_equal(F.grid.lon, self.reference.grid.lon[[0, 5]], decimal=5)

    def test_get_multipe_lat(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [29, 34]})
        test.assert_almost_equal(F.grid.lat, self.reference.grid.lat[[29, 34]], decimal=5)

    def test_get_multipe_lonc(self):
        F = FileReader(self.stub.ncfile.name, dims={'nele': [0, 5]})
        test.assert_almost_equal(F.grid.lonc, self.reference.grid.lonc[[0, 5]], decimal=5)

    def test_get_multipe_latc(self):
        F = FileReader(self.stub.ncfile.name, dims={'nele': [29, 34]})
        test.assert_almost_equal(F.grid.latc, self.reference.grid.latc[[29, 34]], decimal=5)

    # def test_get_bounding_box(self):
    #     wesn = [-5, -3, 50, 55]
    #     extents = [-4.9847326278686523, -3.0939722061157227,
    #                50.19110107421875, 54.946651458740234]
    #     F = FileReader(self.stub.ncfile.name, dims={'wesn': wesn})
    #     test.assert_equal(F.grid.lon.min(), extents[0])
    #     test.assert_equal(F.grid.lon.max(), extents[1])
    #     test.assert_equal(F.grid.lat.min(), extents[2])
    #     test.assert_equal(F.grid.lat.max(), extents[3])

    def test_get_water_column(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [5], 'time': 10}, variables=['temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[10, :, 5], decimal=5)

    def test_get_time_series(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': np.arange(10, 40)}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), self.reference.data.zeta[10:40, 10], decimal=5)

    def test_get_negative_time_series(self):
        F1 = FileReader(self.stub.ncfile.name, dims={'node': [10]}, variables=['zeta'])
        F2 = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': -np.arange(10, 40)}, variables=['zeta'])
        test.assert_almost_equal(F2.data.zeta, F1.data.zeta[-np.arange(10, 40)], decimal=5)

    def test_get_single_time(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [10]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), self.reference.data.zeta[10, 10], decimal=5)

    def test_get_single_time_negative_index(self):
        F = FileReader(self.stub.ncfile.name, dims={'node': [10], 'time': [-10]}, variables=['zeta'])
        test.assert_almost_equal(np.squeeze(F.data.zeta), self.reference.data.zeta[-10, 10], decimal=5)

    def test_get_layer(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5]}, variables=['ww'])
        test.assert_almost_equal(np.squeeze(F.data.ww), self.reference.data.ww[:, 5, :], decimal=5)

    def test_get_layer_get_nodes(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'node': np.arange(4)}, variables=['temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[:, 5, :4], decimal=5)

    def test_get_layer_get_nodes_get_elements(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'node': np.arange(4), 'nele': np.arange(3)}, variables=['ww', 'temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[:, 5, :4], decimal=5)
        test.assert_almost_equal(np.squeeze(F.data.ww), self.reference.data.ww[:, 5, :3], decimal=5)

    def test_get_layer_get_level_get_nodes_get_elements(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'siglev': [4], 'node': np.arange(4), 'nele': np.arange(3)}, variables=['ww', 'temp'])
        test.assert_almost_equal(np.squeeze(F.data.temp), self.reference.data.temp[:, 5, :4], decimal=5)
        test.assert_almost_equal(np.squeeze(F.data.ww), self.reference.data.ww[:, 5, :3], decimal=5)

    def test_get_layer_no_variable(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglay': np.arange(0, 10, 2)})
        test.assert_almost_equal(F.grid.siglay, self.reference.grid.siglay[0:10:2])

    def test_get_level_no_variable(self):
        F = FileReader(self.stub.ncfile.name, dims={'siglev': np.arange(0, 11, 2)})
        test.assert_almost_equal(F.grid.siglev, self.reference.grid.siglev[0:11:2])

    def test_non_temporal_variable(self):
        h = np.asarray([1.64808428, 12.75706577, 18.34670639, 24.29236031,
                        29.7772541, 25.00211716, 22.69193077, 18.70510674,
                        21.96312141, 27.35856438, 35.32657623, 32.48567581,
                        38.93023682, 43.63704681, 51.21723175, 53.23581314,
                        59.78393555, 55.53053284, 52.84440994, 57.36302185,
                        62.2620163, 66.50558472, 61.24137878, 60.91600418,
                        67.42472839, 73.38938904, 70.63117981, 70.62969208,
                        75.18034363, 79.09741974, 84.4043808 , 81.0752182,
                        88.22835541, 90.34424591, 97.57055664, 98.27231598,
                        100.0000000, 96.82516479, 91.1933136 , 88.29994202,
                        89.59196472, 91.40013885, 85.90748596, 79.28456879,
                        74.37998199, 70.46596527, 70.78884888, 70.06604004,
                        63.42258453, 63.06575394, 59.99647141, 57.27880096,
                        55.11286545, 61.5132103, 62.31158066, 59.2288208,
                        53.60129929, 50.73873138, 56.42451477, 52.42653656,
                        44.78648376, 39.55376434, 32.51250839, 28.38024521,
                        20.91413689, 18.19268227, 11.62014961, 7.51470757,
                        38.44644928, 45.77177048, 34.9041214, 51.38194275,
                        77.87741852, 81.04411316])
        F = FileReader(self.stub.ncfile.name, variables=['h'])
        test.assert_almost_equal(F.data.h, h)

    def test_non_temporal_variable_with_dimension(self):
        h = np.asarray([1.64808428, 12.75706577, 18.34670639, 24.29236031])
        F = FileReader(self.stub.ncfile.name, variables=['h'], dims={'node': np.arange(4)})
        test.assert_almost_equal(F.data.h, h)

    def test_add_files(self):
        # Make another stub file which follows in time from the existing one. Then only load a section of that in
        # time and make sure the results are the same as if we'd loaded them manually and added them together.
        next_stub = StubFile(self.endtime, self.endtime + relativedelta(months=1), self.interval,
                             lon=self.lon, lat=self.lat, triangles=self.triangles, zone='30N')

        # Append the new stub file to the old one.
        F1 = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F2 = FileReader(next_stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        all_times = np.concatenate((F1.time.datetime[:], F2.time.datetime[:]), axis=0)
        all_data = np.concatenate((F1.data.ww[:], F2.data.ww[:]), axis=0)
        # Repeat the process, but use the __add__ method in FileReader.
        F1 = FileReader(self.stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F2 = FileReader(next_stub.ncfile.name, dims={'siglay': [5], 'time': [0, -10]}, variables=['ww'])
        F = F2 >> F1

        test.assert_equal(F.time.datetime, all_times)
        test.assert_equal(F.data.ww, all_data)

    def test_get_time_with_string(self):
        time_dims = ['2001-02-12 09:00:00.00000', '2001-02-14 12:00:00.00000']
        returned_indices = np.arange(26, 77)

        F = FileReader(self.stub.ncfile.name, dims={'time': time_dims})
        test.assert_equal(F._dims['time'], returned_indices)

    def test_get_time_with_datetime(self):
        time_dims = [datetime.strptime('2001-02-12 09:00:00.00000', '%Y-%m-%d %H:%M:%S.%f'),
                     datetime.strptime('2001-02-14 12:00:00.00000', '%Y-%m-%d %H:%M:%S.%f')]
        returned_indices = [26, 76]

        F = FileReader(self.stub.ncfile.name, dims={'time': time_dims})
        test.assert_equal(F._dims['time'][0], returned_indices[0])
        test.assert_equal(F._dims['time'][-1], returned_indices[-1])

    def test_get_time_with_tolerance(self):
        time_dims = [datetime.strptime('2001-02-12 09:00:00.00000', '%Y-%m-%d %H:%M:%S.%f'),
                     datetime.strptime('2001-02-12 09:14:02.00000', '%Y-%m-%d %H:%M:%S.%f')]
        returned_indices = [None, 26]

        F = FileReader(self.stub.ncfile.name)
        file_indices = [F.time_to_index(i, tolerance=10) for i in time_dims]
        test.assert_equal(file_indices, returned_indices)