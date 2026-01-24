import subprocess

def unity_runner():

    unity_build_run = r"C:\Users\kmu61\OneDrive\Desktop\Hackbot\unity_sim\My project (1).exe"

    subprocess.Popen([unity_build_run])

    print("process is complete")