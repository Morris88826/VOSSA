import os
import pytorch_lightning as pl
import pyrootutils

pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)
from src.utils import CustomLightningCLI  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("DEBUG", False):
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    pl.seed_everything(1234, workers=True)
    cli = CustomLightningCLI()
