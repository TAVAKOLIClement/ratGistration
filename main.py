import pipelines

if __name__ == "__main__":

    above_folder = "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Above_Acquisition\\"
    below_folder = "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Below_Acquisition\\"
    pipelines.bothSkullExtractionPipeline(above_folder, below_folder, "Au")