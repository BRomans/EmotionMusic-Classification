
class Participant:

    def __init__(self, name):
        self.name = name
        self.approach_withdrawal_idx = []
        self.frontal_midline_theta_idx = []
        self.prefrontal_alpha_pw = []
        self.r_parietal_alpha_pw = []
        self.alpha_parietal = []
        self.asymmetries_alpha = []
        self.beta_fz = []
        self.channels_db = []
        self.channels_z = []
        self.channels_z_cb = []
        self.theta_fz = []

    def set_alpha_parietal(self, alpha_parietal):
        self.alpha_parietal = alpha_parietal

    def set_asymmetries_alpha(self, asymmetries_alpha):
        self.asymmetries_alpha = asymmetries_alpha

    def set_beta_fz(self, beta_fz):
        self.beta_fz = beta_fz

    def set_channels_db(self, channels_db):
        self.channels_db = channels_db

    def set_channels_z(self, channels_z):
        self.channels_z = channels_z

    def set_channels_z_cb(self, channels_z_cb):
        self.channels_z_cb = channels_z_cb

    def set_theta_fz(self, theta_fz):
        self.theta_fz = theta_fz

    def get_alpha_parietal(self):
        return self.alpha_parietal

    def get_asymmetries_alpha(self):
        return self.asymmetries_alpha

    def get_beta_fz(self):
        return self.beta_fz

    def get_channels_db(self):
        return self.channels_db

    def get_channels_z(self):
        return self.channels_z

    def get_channels_z_cb(self):
        return self.channels_z_cb

    def get_theta_fz(self):
        return self.theta_fz


