from model.baseline import CustomCLIP
from model.ThreeBranch import ThreeBranch
from model.LADER import LADER
# Other models
from model.others.MSCI import MSCI
from model.others.Troika import Troika
from model.others.TroikaLADER import TroikaLADER


def get_model(config, attributes, classes, offset, logger, dtype=None, device='cuda:0'):
    if config.model_name == 'ThreeBranch':
        model = ThreeBranch(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'LADER':
        model = LADER(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'Baseline':
        model = CustomCLIP(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    # others methods
    elif config.model_name == 'Troika':
        model = Troika(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'TroikaLADER':
        model = TroikaLADER(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'MSCI':
        model = MSCI(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )
    model = model.to(device)
    return model
