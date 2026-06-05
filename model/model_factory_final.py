
import torch


def get_model(config, attributes, classes, offset, logger, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model_name == 'BAD':
        from model.PCI.BAD import BAD
        model = BAD(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'BADwoOT':
        from model.ablation.BADwoOT import BADwoOT
        model = BADwoOT(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'PCM':
        from model.PCI.PCM import PCM
        model = PCM(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'Troika':
        from model.others.troika import Troika
        model = Troika(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'MSCI':
        from model.others.msci import MSCI
        model = MSCI(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'LADER':
        from model.LADER.LADER import LADER
        model = LADER(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name in {'LADER_MLP', 'LADER-MLP'}:
        from model.ablation.Decoder.MLP import LADERMLP
        model = LADERMLP(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name in {'LADER_Complex', 'LADER-Complex'}:
        from model.ablation.Decoder.Complex import LADERComplex
        model = LADERComplex(config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    elif config.model_name == 'SigLIP_ZeroShot_Baseline':
        from model.baseline.baseline_siglip import SigLIP_ZeroShot_Baseline
        model = SigLIP_ZeroShot_Baseline(
            config, attributes=attributes, classes=classes, offset=offset, device=device, logger=logger)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )
    model = model.to(device)
    return model
