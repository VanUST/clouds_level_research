# /main.py
import config
from src.camera import StereoCameraSystem
from src.pipeline import PipelineRunner

def main():
    """
    Main entry point for the cloud distance measurement pipeline.
    
    This function reads the configuration, initializes the stereo camera system
    and pipeline runner, and executes the task specified in the config file.
    """
    print(f"Starting task: {config.RUN_TASK}")

    # 1. Initialize the Stereo Camera System with parameters from config
    stereo_system = StereoCameraSystem(
        base=config.STEREO_BASE,
        angle_of_view=config.ANGLE_OF_VIEW,
        image_width=config.IMAGE_WIDTH,
        model=config.CAMERA_MODEL
    )

    # 2. Initialize the Pipeline Runner
    runner = PipelineRunner(stereo_system, config)

    # 3. Execute the specified task
    if config.RUN_TASK == 'PROCESS_DIR':
        runner.process_directory()
        
    elif config.RUN_TASK == 'ITERATIVE_REFINEMENT':
        runner.run_iterative_refinement()
        
    elif config.RUN_TASK == 'ABLATE_HOMOGRAPHY':
        runner.run_homography_ablation()
    
    elif config.RUN_TASK == 'PROCESS_TRAIL':
        runner._process_trail_pair()
        
    elif config.RUN_TASK == 'PROCESS_TIME_SERIES':
        runner.process_time_series()
        
    else:
        print(f"Error: Unknown task '{config.RUN_TASK}' specified in config.py.")
        print("Available tasks: 'PROCESS_DIR', 'ITERATIVE_REFINEMENT', 'ABLATE_HOMOGRAPHY', 'PROCESS_TRAIL', 'PROCESS_TIME_SERIES'")

if __name__ == "__main__":
    main()