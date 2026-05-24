from datetime import datetime
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from diffusers import UNet2DModel, DDIMScheduler
from diffusers import DDPMPipeline, DiffusionPipeline
from numpy import ndarray
from scipy.stats import beta


class Task:
    def __init__(self, version:int=1):
        self.path:str = "/home/svu/e0556748"
        self.file_path:str = f"/home/svu/e0556748/res_output_{version}.txt"

    def append_line(self, line_to_append):
        with open(self.file_path, 'a') as file:
            file.write(line_to_append + '\n')

    def estimate_klub(self, train_loader, denoiser, input_var: list, monte_carlo_sample: int, device):
        def p(t_proposed, t_mid, c=0.5):
            return (1 / t_proposed ** 3) * (1 / (t_proposed ** 2 + c ** 2) - 1 / (t_mid ** 2 + c ** 2))

        def importance_sample(t_min: float, t_mid: float, t_max: float, n_samples=1000):
            """
            Importance sampling for t according to π(t) ∝ 1/t^3 * (1/(t^2 + c^2) - 1/(t_mid^2 + c^2)).
            Uses custom proposal distribution.
            """
            # Generate samples from two Beta distributions
            print(t_min, t_mid, t_max)
            a1, b1 = 2, 5  # Parameters for the first Beta distribution
            a2, b2 = 5, 2  # Parameters for the second Beta distribution

            # Generate samples for the first interval [t_min, t_mid]
            n_samples1 = n_samples // 2
            x_samples1 = beta.rvs(a1, b1, size=n_samples1)
            x_samples1 = t_min + (t_mid - t_min) * x_samples1

            # Generate samples for the second interval [t_mid, t_max]
            n_samples2 = n_samples - n_samples1
            x_samples2 = beta.rvs(a2, b2, size=n_samples2)
            x_samples2 = t_mid + (t_max - t_mid) * x_samples2

            # Combine the samples
            x_samples = np.concatenate([x_samples1, x_samples2])

            # Apply a transformation to better match the target distribution
            weights = p(x_samples, t_mid, c)

            # Ensure weights are non-negative
            weights = np.clip(weights, 0, None)
            if np.sum(weights) == 0:
                return np.random.choice(x_samples)

            # Normalize the weights
            print("The weights are {}".format(weights))
            weights /= np.sum(weights)

            # Sample one value according to the weights
            index = np.random.choice(np.arange(len(x_samples)), p=weights)
            return x_samples[index]

        t_min = input_var[0]
        t_mid = input_var[1]
        t_max = input_var[2]
        c = 0.5
        KLUB = [0] * monte_carlo_sample
        for i in range(0, monte_carlo_sample, 1):
            x_0, _ = next(iter(train_loader))
            x_0 = x_0.to(device)
            t = torch.tensor(importance_sample(t_min, t_mid, t_max), device=device)


            t_upper = torch.tensor(t_mid if t < t_mid else t_max, device=device)
            x_t = x_0 + t * torch.randn_like(x_0).to(device)
            x_t_upper = x_t + torch.sqrt((t_upper ** 2 - t ** 2).clone().detach()) * torch.randn_like(x_0).to(
                device)

            # denoised_t = denoiser.unet(x_t, t).sample
            # denoised_t_upper = denoiser.unet(x_t_upper, t_upper).sample
            denoiser.scheduler.set_timesteps(1000)
            denoised_t = denoiser.unet(x_t, t).sample
            denoised_t_upper = denoiser.unet(x_t_upper, t_upper).sample

            diff = denoised_t - denoised_t_upper

            reweighting_factor = 1 / (1 / (t ** 2 + c ** 2) - 1 / (t_upper ** 2 + c ** 2))
            klub_est = (torch.norm(diff, p=2) ** 2) * reweighting_factor

            KLUB[i] = klub_est.item()
        return np.mean(KLUB)

    def klub_optimization_algo_one(self, denoiser, schedule, train_loader, monte_carlo_sample, device):
        print("Start the AYS")
        noChange: bool = False
        n: int = len(schedule) - 1   # n = 10  while schedule includes t_0 to t_10
        n_candidates: int = 10
        r: int = n + 1 # r = 11
        round: int = 0

        def get_candidates(left_neigh: float, right_neigh: float, current: float, num: int):
            candidates = np.linspace(left_neigh, right_neigh, num+1, False).tolist()
            candidates = np.append(candidates, current)
            candidates = candidates[1:]
            return candidates

        #  TODO Check the Non-adj Method
        while not noChange:
            noChange = True
            for i in range(1, n):
                candidates = get_candidates((schedule[i-1] + schedule[i + 1])/2, (schedule[i] + schedule[i + 1]) / 2, schedule[i], n_candidates)
                monitor = f"[Monitor] Round {round} candidates: {candidates}"
                self.append_line(monitor)
                KLUB = [0] * (r)
                for j in range(0, r):
                    KLUB[j] = self.estimate_klub(train_loader, denoiser, [schedule[i - 1], candidates[j], schedule[i + 1]],
                                            monte_carlo_sample, device)
                    klub_monitor = f"Round {round} KLUB: {KLUB[j]}"
                    self.append_line(klub_monitor)
                minIdx = np.argmin(KLUB)
                if candidates[minIdx] != schedule[i]:
                    schedule[i] = candidates[minIdx]
                    noChange = False


        # while not noChange:
        #     noChange = True
        #     for i in range(1, n):
        #         candidates = get_candidates(schedule[i - 1], schedule[i + 1], schedule[i], n_candidates)
        #         monitor = f"[Monitor] Round {round} candidates: {candidates}"
        #         self.append_line(monitor)
        #         KLUB = [0] * (r)
        #         for j in range(0, r):
        #             KLUB[j] = self.estimate_klub(train_loader, denoiser, [schedule[i - 1], candidates[j], schedule[i + 1]],
        #                                     monte_carlo_sample, device)
        #             klub_monitor = f"Round {round} KLUB: {KLUB[j]}"
        #             self.append_line(klub_monitor)
        #         minIdx = np.argmin(KLUB)
        #         if candidates[minIdx] != schedule[i]:
        #             schedule[i] = candidates[minIdx]
        #             noChange = False

            print(f"Round {round} res: {schedule} {datetime.now().strftime('%H:%M:%S')}")
            self.append_line(f"Round {round} res: {schedule} {datetime.now().strftime('%H:%M:%S')}")
            round += 1

    def run(self, initial_schedule:ndarray=None):
        # Initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model_id = "google/ddpm-cifar10-32"

        denoiser = DDPMPipeline.from_pretrained(model_id)
        denoiser.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
            set_alpha_to_one=False
        )
        denoiser.scheduler.alphas_cumprod = denoiser.scheduler.alphas_cumprod.to(device)
        denoiser.to(device)
        denoiser.unet.eval()
        print("Denoiser loaded successfully!")

        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        if initial_schedule is None:
            initial_schedule = np.linspace(0, 1000, 11)

        self.klub_optimization_algo_one(denoiser, initial_schedule, train_loader, 10, device)


task = Task(10)
# ays = [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0]
# ays_input = list(reversed(ays))
# initial_schedule = np.array(ays_input)
task.run()





