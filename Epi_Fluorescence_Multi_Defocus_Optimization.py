import queue
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import CubicSpline
from Misc_Tools import normalization
from DM_Control_Modules import AlPaoDM
from Main_Camera_Control_Modules import MainCamera
from Zernike_Polynomials_Modules import mean_gradient, input_zernike

CONFIG = {
    "OPTIMIZATION" : {
        "defocus_steps" : 11,
        "defocus_amp" : 0.9,
        "zernike_amp" : 0.6,
        "max_order" : 13,
        "lhs_sampling": 160,
        "spgd_iter": 64,
        "spgd_gamma_init": 0.1,
        "spgd_gamma_decay": 0.01,
        "momentum_gamma": 0.05,
        "disturb_gamma": 0.5,
        "jumper_window": 8,
        "jumper": 0.05,
        "normalize_frame" : False
    },
    "CAMERA" : {
        "camera_size" : 2304,
        "subarray_mode" : 2.0,
        "subarray_size" : (1024, 1024),
        "sensor_mode" : 12.0,
        "exposure_time" : 40 * 1e-6,
        "internal_line_interval" : 1 * 1e-6,
        "trigger_source" : 1
    },
    "VISUALIZATION" : {
        "refresh_fps" : 20,
        "colormap" : "viridis",
        "disp_range" : [0, 2**16 - 1],
        "max_metric_queue": 160
    }
}

class EpiFluorescenceOptimization:

    def __init__(self):

        self.camera = MainCamera()
        self.dm = AlPaoDM()

        self.frame_queue = queue.Queue(maxsize = 8)
        self.metric_queue = queue.Queue(maxsize = CONFIG["VISUALIZATION"]["max_metric_queue"])
        self.optim_data = {} # Format: {defocus_amp: {zern_coffs, metric}}

        self.defocus_amps = np.linspace(-CONFIG["OPTIMIZATION"]["defocus_amp"], CONFIG["OPTIMIZATION"]["defocus_amp"],
                                        CONFIG["OPTIMIZATION"]["defocus_steps"])
        self.zernike_indices = [idx for idx, _ in enumerate(input_zernike) if
                                2 <= idx <= CONFIG["OPTIMIZATION"]["max_order"]  and idx != 3]

        self.live_fig = None
        self.stop_event = threading.Event()

    def _live_feed_thread(self):

        plt.ion()
        self.live_fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6.18))
        self.live_fig.tight_layout(pad = 4)

        null_img = np.zeros(CONFIG["CAMERA"]["subarray_size"], dtype = np.uint16)
        img_1 = ax1.imshow(null_img, cmap = CONFIG["VISUALIZATION"]["colormap"],
                           vmin = CONFIG["VISUALIZATION"]["disp_range"][0],
                           vmax = CONFIG["VISUALIZATION"]["disp_range"][1]) if not CONFIG["OPTIMIZATION"]["normalize_frame"] else ax1.imshow(null_img, cmap = CONFIG["VISUALIZATION"]["colormap"], vmin = -1, vmax = 1)

        ax1.set_title("Camera Feed")

        img2, = ax2.plot([], [], "b-")
        ax2.set_title("Metric Trend")
        ax2.set_xlabel("Frame NO.")
        ax2.set_ylabel("Metric History")

        metric_trend = []
        last_update_time = time.time()

        print("Live feed initiated...")

        while not self.stop_event.is_set():
            try:

                current_frame = self.frame_queue.get_nowait()
                if CONFIG["OPTIMIZATION"]["normalize_frame"]:
                    current_frame = normalization(current_frame)
                print(np.max(current_frame))
                img_1.set_data(current_frame)
                # img_1.autoscale()

            except queue.Empty:
                pass

            except queue.Full:
                self.frame_queue.get_nowait()
                pass

            try:

                current_metric = self.metric_queue.get_nowait()
                # print(current_metric)
                metric_trend.append(current_metric)

                if len(metric_trend) > CONFIG["VISUALIZATION"]["max_metric_queue"]:
                    metric_trend.pop(0)

                img2.set_data(np.arange(len(metric_trend)), metric_trend)
                ax2.relim()
                ax2.autoscale_view()

            except queue.Empty:
                pass

            except queue.Full:
                self.metric_queue.get_nowait()
                pass

            if time.time() - last_update_time > 1/CONFIG["VISUALIZATION"]["refresh_fps"]: # FPS
                self.live_fig.canvas.flush_events()
                last_update_time = time.time()

    def _lhs_sampling(self, num_samples):

        samples = np.zeros((num_samples, len(self.zernike_indices)))
        for i in range(len(self.zernike_indices)):
            edges = np.linspace(-1, 1, num_samples + 1)
            samples[:, i] = np.random.permutation(
                np.random.uniform(edges[:-1], edges[1:])
            )
        print("Performing lhs sampling...")
        return samples

    def _spgd_core(self, defocus_amp, initial_coffs):

        current_coffs = initial_coffs.copy()
        best_metric = -np.inf
        best_coffs = None
        momentum = np.zeros_like(current_coffs)
        metric_trend = []

        try:
            print(f"Performing spgd optimization at defocus: {defocus_amp:.2f}")
            for k in range(CONFIG["OPTIMIZATION"]["spgd_iter"]):

                disturb = np.random.normal(
                    scale = CONFIG["OPTIMIZATION"]["disturb_gamma"],
                    size = current_coffs.shape
                )

                coff_plus = current_coffs + disturb
                metric_plus = self._eval_metric(defocus_amp, coff_plus)

                coff_minus = current_coffs - disturb
                metric_minus = self._eval_metric(defocus_amp, coff_minus)

                delta_metric = metric_plus - metric_minus
                gradient = delta_metric * disturb
                momentum = CONFIG["OPTIMIZATION"]["momentum_gamma"] * momentum + (1 - CONFIG["OPTIMIZATION"]["momentum_gamma"] * gradient)
                gamma = CONFIG["OPTIMIZATION"]["spgd_gamma_init"] * np.exp(-CONFIG["OPTIMIZATION"]["spgd_gamma_decay"] * k)
                current_coffs += gamma * momentum

                current_metric = self._eval_metric(defocus_amp, current_coffs)
                metric_trend.append(current_metric)
                self.metric_queue.put(current_metric)

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_coffs = current_coffs.copy()

                if len(metric_trend) > 2 * CONFIG["OPTIMIZATION"]["jumper_window"]:
                    recent_gain = np.mean(metric_trend[-CONFIG["OPTIMIZATION"]["jumper_window"], :],
                                          metric_trend[-2*CONFIG["OPTIMIZATION"]["jumper_window"]:-CONFIG["OPTIMIZATION"]["jumper_window"]])
                    if abs(recent_gain) < 0.01 * best_metric:
                        print("spgd jumper quit...")
                        break
                return best_coffs, best_metric

        except Exception as e:
            print(f"Error in spgd core: {str(e)}")
            raise


    def _eval_metric(self, defocus_amp, zern_coffs):
        try:

            full_coffs = np.zeros((len(input_zernike), 1))
            full_coffs[3] = defocus_amp
            full_coffs[self.zernike_indices] = zern_coffs.reshape(-1, 1)

            self.dm.send_direct_zernike(full_coffs)
            time.sleep(0.01)

            frame = self._acquire_frame()
            self.frame_queue.put(frame)

            return mean_gradient(frame)

        except queue.Full:
            self.frame_queue.get()
            return 0.0

        except Exception as e:
            print(f"Error in eval metric: {str(e)}")
            raise

    def hybrid_spgd_optimize(self):

        self._setup_camera()
        self.dm.start_direct_control()

        live_thread = threading.Thread(target = self._live_feed_thread)
        live_thread.daemon = True;live_thread.start()

        try:
            for defocus_amp in self.defocus_amps:

                defocus_carrier = np.zeros((len(input_zernike), 1))
                defocus_carrier[3] = defocus_amp

                self.dm.send_direct_zernike(defocus_carrier)
                input(f"Now at defocus amp: {defocus_amp:.2f}, please refocus.")

                lhs_guess = self._lhs_sampling(CONFIG["OPTIMIZATION"]["lhs_sampling"])
                best_lhs_metric = -np.inf
                best_lhs_coffs = None

                for lhs_coff in lhs_guess:
                    metric = self._eval_metric(defocus_amp, lhs_coff)
                    if metric > best_lhs_metric:
                        best_lhs_metric = metric
                        best_lhs_coffs = lhs_coff.copy()

                best_global_coffs, best_global_metric = self._spgd_core(defocus_amp, best_lhs_coffs)

                self.optim_data[defocus_amp] = {
                    "coffs": best_global_coffs,
                    "metric": best_global_metric
                }

        except KeyboardInterrupt:
            print("User interrupt...")

        except Exception as e:
            print(f"Error in hybrid optimize: {str(e)}")

        finally:
            self.stop_event.set()
            live_thread.join()
            time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
            np.save(f"C:\Xiong_Jianxuan\Python_Projects\Adaptive_Optics\Data_Deposit\\optimum_coffs_{str(time_stamp)}.npy", self.optim_data)
            print("Completed, releasing resources and saving data...")
            self.camera.camera_close()
            self.dm.stop_loop()

    # def _compare_update(self, defocus_amp):
    #     if self.compare_fig is None:
    #         plt.ion()
    #         self.compare_fig, self.compare_axes = plt.subplots(2, 1, figsize = (10, 6.18))
    #         self.compare_fig.tight_layout()
    #
    #
    #     rows = len(self.zernike_indices)
    #     cols = 2
    #     axes = self.compare_fig.subplots(rows, cols)
    #
    #     for row, (zern, data) in enumerate(self.optim_data.items()):
    #
    #         axes[row, 0].imshow(data["before"][defocus_amp], cmap = CONFIG["VISUALIZATION"]["colormap"])
    #         axes[row, 0].set_title(f"Order {zern} Before")
    #
    #         axes[row, 1].imshow(data["after"][defocus_amp], cmaps = CONFIG["VISUALIZATION"]["colormap"])
    #         axes[row, 1].set_title(f"Order {zern} After")
    #
    #     self.compare_fig.tight_layout()
    #     self.compare_fig.canvas.draw()

    # def loop_optimize(self):
    #     try:
    #
    #         self._setup_camera()
    #         self.dm.start_direct_control()
    #         live_thread = threading.Thread(target=self._live_feed_thread)
    #         live_thread.daemon = True;live_thread.start()
    #
    #     try:
    #
    #         null_img = np.zeros((CONFIG["CAMERA"]["subarray_size"]), dtype = np.uint16)
    #         self.frame_queue.put_nowait(null_img)
    #         self.metric_queue.put(0.)
    #
    #         for defocus_amp in self.defocus_amps:
    #
    #             self._apply_defocus(defocus_amp)
    #
    #             input(f"Now at defocus amp: {defocus_amp:.2f}, please refocus.")
    #
    #             self._loop_core(defocus_amp)
    #
    #             self._compare_update(defocus_amp)
    #
    #     finally:
    #         self.stop_event.set()
    #         live_thread.join()
    #         self._save_results()
    #
    # def _loop_core(self, defocus_amp):
    #
    #     initial_frame = self._acquire_frame()
    #     initial_metric = self._calc_metric(initial_frame)
    #     optimal_amps = {order: 0. for order in self.target_zernikes}
    #
    #     for zern_order in self.target_zernikes:
    #
    #         metrics = []
    #         frames = []
    #
    #         for amp in self.zernike_amps:
    #
    #             self._apply_zernike(zern_order, amp, defocus_amp)
    #             frame = self._acquire_frame()
    #             metric = self._calc_metric(frame)
    #             d_metric = metric - initial_metric
    #
    #             metrics.append(d_metric)
    #             frames.append(frame)
    #
    #             self.frame_queue.put(frame)
    #             self.metric_queue.put(metric)
    #
    #         optimal_amp = self._find_optimal_amp(metrics)
    #         optimal_amps[zern_order] = optimal_amp
    #         self._cache_results(zern_order, defocus_amp, frames, optimal_amp)
    #
    # def _apply_defocus(self, amp):
    #     zernike = np.zeros((len(input_zernike), 1))
    #     zernike[3] = amp
    #     self.dm.send_direct_zernike(zernike.T)
    #
    # def _apply_zernike(self, zern_order, amp, defocus_amp):
    #     zernike = np.zeros((len(input_zernike), 1))
    #     zernike[3] = defocus_amp
    #     zernike[zern_order] = amp
    #     self.dm.send_direct_zernike(zernike.T)

    def _acquire_frame(self):
        try:
            frame = self.camera.get_last_live_frame()
            if frame is not None:
                return frame
        except Exception as e:
            print(f"Error in acquiring frame: {e}")

    # def _cache_results(self, zern_order, defocus_amp, frames, optimal_amp):
    #     if zern_order not in self.optim_data:
    #         self.optim_data[zern_order] = {
    #             "before": [],
    #             "after": []
    #         }

    def _setup_camera(self):
        try:

            self.camera.camera_open()
            self.camera.set_single_parameter("subarray_mode", CONFIG["CAMERA"]["subarray_mode"])
            self.camera.set_single_parameter("subarray_hsize", CONFIG["CAMERA"]["subarray_size"][0])
            self.camera.set_single_parameter("subarray_vsize", CONFIG["CAMERA"]["subarray_size"][1])
            self.camera.set_single_parameter("subarray_hpos",
                                             int((CONFIG["CAMERA"]["camera_size"] / 2 - CONFIG["CAMERA"]["subarray_size"][
                                                 0] / 2)))
            self.camera.set_single_parameter("subarray_vpos",
                                             int((CONFIG["CAMERA"]["camera_size"] / 2 - CONFIG["CAMERA"]["subarray_size"][
                                                 1] / 2)))
            self.camera.set_single_parameter("sensor_mode", CONFIG["CAMERA"]["sensor_mode"])
            self.camera.set_single_parameter("exposure_time", CONFIG["CAMERA"]["exposure_time"])
            self.camera.set_single_parameter("internal_line_interval", CONFIG["CAMERA"]["internal_line_interval"])
            self.camera.set_single_parameter("trigger_source", CONFIG["CAMERA"]["trigger_source"])
            self.camera.start_live()
            print("Camera initiated...")

        except Exception as e:
            print(f"Error in opening camera: {e}")

    @staticmethod
    def _find_optimal_amp(metric_trend):

        x = np.linspace(-CONFIG["OPTIMIZATION"]["zernike_amp"], CONFIG["OPTIMIZATION"]["zernike_amp"], len(metric_trend))
        y = np.array(metric_trend)

        spline = CubicSpline(x, y, bc_type="natural")
        x_fine = np.linspace(-CONFIG["OPTIMIZATION"]["zernike_amp"], CONFIG["OPTIMIZATION"]["zernike_amp"],
                             CONFIG["OPTIMIZATION"]["interp_num"])
        y_fine = spline(x_fine)

        peak_idx = np.argmax(y_fine)
        return np.float64(x_fine[peak_idx])

if __name__ == '__main__':
    optimizer = EpiFluorescenceOptimization()
    optimizer.hybrid_spgd_optimize()