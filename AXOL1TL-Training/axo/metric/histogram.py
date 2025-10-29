import numpy as np
import h5py

class distribution_plots:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.background_hist = None
        self.signal_hist = {}
        self.pu_hist = {}
        self.ht_hist = {}
        self.object_pt_hist = {}
        self.data_file = None
        self._open_hdf5()
        ap_fixed = self.config["precision"]
        self.make_dist()

    def _open_hdf5(self):
        self.data_file = h5py.File(self.config["data_path"], "r")

    def make_dist(self):
        pu_bins = np.arange(0, 71, 5)

        x_test = self.data_file["Background_data"]["Test"]["DATA"][:]
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_bg = self.model.predict(x_test, batch_size=120000, verbose=0).ravel()
        ht_bg = self.data_file["Background_data"]["Test"]["HT"][:]

        # Background histogram (1D) for the score
        self.background_hist = np.histogram(y_bg, bins=200)

        # HT histogram for background, if desired
        self.ht_hist["background"] = np.histogram2d(y_bg, ht_bg, bins=[200, 50])

        # We do NOT fill self.pu_hist["background"] here, so there's no 2D PU histogram for background
        # The object pT histogram remains the same
        self.object_pt_hist["background"] = {
            "eg_leading": np.histogram2d(
                y_bg, np.sort(x_test[:, 1:13][:, ::2], axis=1)[:, -1],
                bins=[200, 50]),
            "eg_subleading": np.histogram2d(
                y_bg, np.sort(x_test[:, 1:13][:, ::2], axis=1)[:, -2],
                bins=[200, 50]),
            "mu_leading": np.histogram2d(
                y_bg, np.sort(x_test[:, 13:21][:, ::2], axis=1)[:, -1],
                bins=[200, 50]),
            "mu_subleading": np.histogram2d(
                y_bg, np.sort(x_test[:, 13:21][:, ::2], axis=1)[:, -2],
                bins=[200, 50]),
            "jet_leading": np.histogram2d(
                y_bg, np.sort(x_test[:, 21:33][:, ::2], axis=1)[:, -1],
                bins=[200, 50]),
            "jet_subleading": np.histogram2d(
                y_bg, np.sort(x_test[:, 21:33][:, ::2], axis=1)[:, -2],
                bins=[200, 50])
        }

        # Now for signals
        for signal in self.data_file["Signal_data"].keys():
            data = self.data_file["Signal_data"][signal]["DATA"][:]
            data = data.reshape(data.shape[0], -1)
            y_sig = self.model.predict(data, batch_size=120000, verbose=0).ravel()
            pu_sig = self.data_file["Signal_data"][signal]["PU"][:]
            ht_sig = self.data_file["Signal_data"][signal]["HT"][:]

            self.signal_hist[signal] = np.histogram(y_sig, bins=200)
            # Fill the PU histogram only for signals
            self.pu_hist[signal] = np.histogram2d(y_sig, pu_sig, bins=[200, pu_bins])
            self.ht_hist[signal] = np.histogram2d(y_sig, ht_sig, bins=[200, 50])
            self.object_pt_hist[signal] = {
                "eg_leading": np.histogram2d(
                    y_sig, np.sort(data[:, 1:13][:, ::2], axis=1)[:, -1],
                    bins=[200, 50]),
                "eg_subleading": np.histogram2d(
                    y_sig, np.sort(data[:, 1:13][:, ::2], axis=1)[:, -2],
                    bins=[200, 50]),
                "mu_leading": np.histogram2d(
                    y_sig, np.sort(data[:, 13:21][:, ::2], axis=1)[:, -1],
                    bins=[200, 50]),
                "mu_subleading": np.histogram2d(
                    y_sig, np.sort(data[:, 13:21][:, ::2], axis=1)[:, -2],
                    bins=[200, 50]),
                "jet_leading": np.histogram2d(
                    y_sig, np.sort(data[:, 21:33][:, ::2], axis=1)[:, -1],
                    bins=[200, 50]),
                "jet_subleading": np.histogram2d(
                    y_sig, np.sort(data[:, 21:33][:, ::2], axis=1)[:, -2],
                    bins=[200, 50])
            }
