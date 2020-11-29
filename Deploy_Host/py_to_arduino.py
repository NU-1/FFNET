import time
from pySerialTransfer import pySerialTransfer as txfer

link = txfer.SerialTransfer('/dev/cu.usbmodem141201')

link.open()
time.sleep(2) # allow some time for the Arduino to completely reset

while True:
    send_size = 0
    
    #send a float
    float_ = 5.234
    float_size = link.tx_obj(float_, send_size) - send_size
    send_size += float_size
    
    #transmit float
    link.send(send_size)
    
    #wait for response
    while not link.available():
        if link.status < 0:
            if link.status == txfer.CRC_ERROR:
                print('ERROR: CRC_ERROR')
            elif link.status == txfer.PAYLOAD_ERROR:
                print('ERROR: PAYLOAD_ERROR')
            elif link.status == txfer.STOP_BYTE_ERROR:
                print('ERROR: STOP_BYTE_ERROR')
            else:
                print('ERROR: {}'.format(link.status))
    
    
    
    #parse float
    rec_float_ = link.rx_obj(obj_type=type(float_),
                             obj_byte_size=float_size,
                             start_pos=(0))
    

    #display data
    print('SENT: {}'.format(float_))
    print('RCVD: {} '.format(rec_float_))
    print(' ')

try:
    link.close()
except:
    pass
