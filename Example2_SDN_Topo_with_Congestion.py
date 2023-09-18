from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.util import irange

class CustomTopo(Topo):
    "Custom topology with congestion"

    def build(self):
        # Add switches and hosts
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')

        # Add links with congestion
        self.addLink(s1, s2, bw=10, delay='50ms', loss=10, max_queue_size=1000, use_htb=True)
        self.addLink(h1, s1, bw=10, delay='50ms', max_queue_size=1000, use_htb=True)
        self.addLink(h2, s2, bw=10, delay='50ms', max_queue_size=1000, use_htb=True)

def create_network():
    "Create custom network with congestion"

    topo = CustomTopo()
    net = Mininet(topo=topo, link=TCLink)

    # Start the network
    net.start()

    # Test network connectivity
    net.pingAll()

    # Start traffic flow with iperf
    h1, h2 = net.get('h1', 'h2')
    h2.cmd('iperf -s -i 1 > iperf_server.txt &')
    h1.cmd('iperf -c %s -i 1 -t 10 > iperf_client.txt &' % h2.IP())

    # Stop the network
    net.stop()

if __name__ == '__main__':
    create_network()






Regenerate response
