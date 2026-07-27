from bitalino import BITalino

class Device():
    def __init__(self, macAddress, samplingRate=1000, acqChannels=[1, 2]):
        self.device = BITalino(macAddress)
        self.samplingRate = samplingRate
        self.acqChannels = acqChannels
        self.macAddress = macAddress

    def read(self, nSamples):
        return self.device.read(nSamples)

    def start(self):
        if self.device:
            self.device.start(self.samplingRate, self.acqChannels)

    def stop(self):
        if self.device:
            self.device.stop()

    def close(self):
        if self.device:
            self.device.close()

    def reset(self):
        try:
            self.device.close()
            self.device.stop()
        except Exception as e:
            print("Erreur lors de la fermeture :", e)
        self.device = BITalino(self.macAddress)
        self.start()
