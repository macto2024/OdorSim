# Connect to GADEN VM via VNC from macOS

This guide covers connecting to the Alibaba Cloud VM (`47.103.195.234`) using VNC, so you can run RViz/GADEN on the VM and view it on your Mac.

---

## 1. Start VNC server on the VM

SSH into the VM (replace with the actual IP if it changes):

```bash
ssh root@47.103.195.234
```

Then start the VNC server:

```bash
vncserver :1 -geometry 1920x1080 -depth 24
```

You should see output like:

```text
New Xtigervnc server 'iZuf6576ey1k47npi9yyzhZ:1 (root)' on port 5901 for display :1.
```

If you get an error saying display `:1` is already in use, kill it first:

```bash
vncserver -kill :1
rm -f /tmp/.X11-unix/X1 /tmp/.X1-lock
vncserver :1 -geometry 1920x1080 -depth 24
```

---

## 2. Open VNC port in Alibaba Cloud security group

If VNC connection fails, open port `5901` in the Alibaba Cloud security group:

1. Log in to [Alibaba Cloud Console](https://www.alibabacloud.com/)
2. Go to **Elastic Compute Service (ECS)** → **Instances**
3. Find your VM and click **Security Group**
4. Click **Add Rule** (Inbound):
   - **Type**: Custom TCP
   - **Port Range**: `5901/5901`
   - **Source**: your Mac IP, or `0.0.0.0/0` for testing
   - **Authorization Object**: `101.5.23.205/32` (or `0.0.0.0/0`)
5. Save the rule.

---

## 3. Connect from macOS

### Option A: Built-in Screen Sharing

1. Open **Finder**
2. Go to **Go → Connect to Server** (or press `Cmd + K`)
3. Enter:

   ```text
   vnc://47.103.195.234:5901
   ```

4. Click **Connect**
5. When prompted, enter the VNC password you set with `vncpasswd`

### Option B: RealVNC Viewer or TigerVNC Viewer

If Finder's Screen Sharing has issues, download a dedicated VNC client:

- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
- [TigerVNC Viewer](https://sourceforge.net/projects/tigervnc/files/stable/)

Open it and connect to:

```text
47.103.195.234:5901
```

Enter the VNC password when prompted.

---

## 4. Run GADEN inside the VNC desktop

Once connected, you will see the XFCE desktop. Open a terminal inside VNC and run:

```bash
cd /root/OdorSim/gaden
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Preprocess (only needed once per scenario)
ros2 launch test_env gaden_preproc_launch.py scenario:=10x6_central_obstacle configuration:=config1

# Run GADEN with RViz
ros2 launch test_env gaden_sim_launch.py scenario:=10x6_central_obstacle configuration:=config1 simulation:=sim1
```

RViz will open inside the VNC desktop and you will see it on your Mac.

---

## 5. After rebooting the VM

If the VM is restarted, VNC does **not** start automatically. Just repeat step 1:

```bash
ssh root@47.103.195.234
vncserver :1 -geometry 1920x1080 -depth 24
```

Then reconnect from your Mac with `vnc://47.103.195.234:5901`.

---

## 6. Optional: make VNC start automatically on boot

To avoid starting VNC manually after every reboot, create a systemd service:

```bash
cat > /etc/systemd/system/vncserver@.service <<'EOF'
[Unit]
Description=Remote desktop service (VNC)
After=syslog.target network.target

[Service]
Type=forking
User=root
ExecStartPre=/bin/sh -c '/usr/bin/vncserver -kill :%i > /dev/null 2>&1 || :'
ExecStart=/usr/bin/vncserver :%i -geometry 1920x1080 -depth 24
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vncserver@1.service
systemctl start vncserver@1.service
```

After this, VNC will start automatically on boot.

---

## Troubleshooting

### VNC session exits immediately

Check the log:

```bash
cat /root/.vnc/iZuf6576ey1k47npi9yyzhZ:1.log
```

Make sure `~/.vnc/xstartup` contains:

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XKL_XMODMAP_DISABLE=1
exec dbus-launch --exit-with-session startxfce4
```

And make it executable:

```bash
chmod +x ~/.vnc/xstartup
```

### Cannot connect from Mac

- Verify VNC is running: `ss -tlnp | grep 5901`
- Verify Alibaba Cloud security group allows inbound TCP on port 5901
- Try connecting with the VM's IP directly from inside the VM to test:
  ```bash
  xtigervncviewer -SecurityTypes VncAuth -passwd ~/.vnc/passwd :1
  ```

### RViz is slow

VNC over the internet can be laggy. Try:
- Lowering the VNC resolution: `vncserver :1 -geometry 1280x720 -depth 16`
- Using a faster VNC client like TigerVNC or RealVNC with low-quality color mode
- Disabling heavy RViz displays (e.g. gas point cloud) once it loads

