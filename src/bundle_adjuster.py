import subprocess

class BundleAdjuster:

    def __init__(self, input_ply, output_ply, bundle_file, ceres_params):
        self.solver = ceres_params['solver']
        self.input_ply = input_ply
        self.output_ply = output_ply
        self.bundle_file = bundle_file
        self.max_iterations = ceres_params['maxIterations']
        self.inner_iterations = ceres_params['inner_iterations']
        self.nonmonotonic_steps = ceres_params['nonmonotonic_steps']

    def bundle_adjust(self):
        subprocess.call([
            self.solver,
            '--input={}'.format(self.bundle_file),
            '--num_iterations={}'.format(self.max_iterations),
            '--inner_iterations={}'.format(self.inner_iterations),
            '--nonmonotonic_steps={}'.format(self.nonmonotonic_steps),
            '--initial_ply={}'.format(self.input_ply),
            '--final_ply={}'.format(self.output_ply),
        ])