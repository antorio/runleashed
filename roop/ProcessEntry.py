class ProcessEntry:
    def __init__(self, filename: str, start: int, end: int, fps: float):
        self.filename = filename
        self.finalname = None
        self.completed = False      # the output file exists and is final (not a partial __temp)
        # set only by the Face Swap tab for a file with a painted keep-original
        # mask (the editor layer); files without keep the run's own mask
        self.startframe = start
        self.endframe = end
        self.fps = fps