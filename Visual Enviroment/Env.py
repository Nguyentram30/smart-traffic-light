import sumo_rl

env = sumo_rl.environment.env.SumoEnvironment(
    net_file='D:\Projects\Python Projects\Visual Enviroment\Network\.net.xml',
    route_file='D:\Projects\Python Projects\Visual Enviroment\Network\route\.rou.xml',
    out_csv_name='outputs/experiment',
    use_gui=True
)
env.render()
