from pydantic import BaseModel


class pm(BaseModel):
    Air_temperature:float
    Process_temperature:float
    Rotational_speed:float
    Torque:float
    Tool_wear: float
    H:int
    L:int
    M:int
    
    
    
