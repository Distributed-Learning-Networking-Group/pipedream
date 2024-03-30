import os


class raw_profile_to_profule():
    def __init__(self, model_name: str) -> None:
        self.dir = 'raw_profile/'
        self.layer_profile = {}
        self._read_raw_profile(model_name)

    def _read_raw_profile(self, model_name: str):
        layer_profile = {}
        with open(self.dir+model_name+'.txt')as file:
            for line in file:
                layer_str = line.split()[0].split('node')[1]
                layer_profile[layer_str] = []
                for data in line.split():
                    if 'forward_compute_time' in data:
                        layer_profile[layer_str].append(
                            float(data.split('=')[1].strip(',')))
                    elif 'backward_compute_time' in data:
                        layer_profile[layer_str].append(
                            float(data.split('=')[1].strip(',')))
                    elif 'activation_size' in data:
                        layer_profile[layer_str].append(
                            float(data.split('=')[1].strip(',')))
                    elif 'parameter_size' in data:
                        layer_profile[layer_str].append(
                            float(data.split('=')[1].strip(',')))
        sorted_keys = sorted(layer_profile, key=lambda x: int(x))
        self.layer_profile = {key: layer_profile[key] for key in sorted_keys}


if __name__ == '__main__':
    test = raw_profile_to_profule('vgg16')
    pass
