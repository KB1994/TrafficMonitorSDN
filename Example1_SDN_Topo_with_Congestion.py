from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.util import irange


class SDN_congestion_topology(Topo):
    "Custom topology with congestion"

    def build(self):
        # Add switches and hosts
        s1=self.addSwitch('s1', protocols=["OpenFlow13"] )
        s2=self.addSwitch('s2', protocols=["OpenFlow13"] )
        h1=self.addHost('h1')
        h2=self.addHost('h2')
        h3=self.addHost('h3')
        h4=self.addHost('h4')

        # Add links with congestion
        

        self.addLink(s1, h1, bw=10, delay='0ms',  loss=0,  max_queue_size=1000, use_htb=True)
        self.addLink(s1, h2, bw=10, delay='0ms', loss=0, max_queue_size=1000, use_htb=True)
        self.addLink(s1, s2, bw=20, delay='0ms', loss=10, max_queue_size=1000, use_htb=True)
        self.addLink(s2, h3, bw=10, delay='0ms', loss=0, max_queue_size=1000, use_htb=True)
        self.addLink(s2, h4, bw=10, delay='0ms', loss=0, max_queue_size=1000, use_htb=True)
        



topos = { 'mytopo': ( lambda: SDN_congestion_topology() ) }
