import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from pyRDDLGym.core.compiler.model import RDDLPlanningModel

"""
Train Route Visualizer for pyRDDLGym.

This class provides a custom 2D visualization of the train system.
It renders:
- A circular route representing the track.
- Stations as squares, with colors indicating passenger crowding levels.
- Trains as circles, positioned dynamically based on their state (In Route, In Queue, At Station).

The visualizer is designed to produce frames for GIF generation during training.
"""
class TrainRouteVisualizer:
    def __init__(self, model):
        self._model = model

        # Rendering configuration
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.route_radius = 0.4
        self.station_size = 0.06
        self.train_size = 0.02
        self.x_center = 0.5
        self.y_center = 0.5
        
        # Load static configuration from RDDL non-fluents
        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        # Pre-process static environment data
        self.drive_time_to_dest = {}
        for k, v in self._nonfluents.items():
            var, objects = RDDLPlanningModel.parse_grounded(k)
            if var == "DRIVE_TIME":
                _, dest = objects
                if v > 0: self.drive_time_to_dest[dest] = v
            if var == "DEPOT_STATION" and v:
                self.depot_station = objects[0]

    def create_route_circle(self):
        """Draws the dashed circular route track."""
        route_circle = plt.Circle(
            (self.x_center, self.y_center),
            self.route_radius,
            color="gray",
            fill=False,
            linestyle="--",
            alpha=0.5
        )
        self.ax.add_patch(route_circle)

    def set_station_color(self, station, state):
        """Returns a color from Red (Full) to Green (Empty) based on passenger count."""
        capacity = 1000
        passengers = state.get(f"passengers_at_station___{station}", 0)
        crowding_level = np.clip(passengers / capacity, 0, 1)

        # RdYlGn_r: Red (High) -> Yellow -> Green (Low) reversed
        cmap = plt.get_cmap("RdYlGn_r")
        return cmap(crowding_level)

    def set_text_color(self, station_color):
        """Ensures station labels are readable against their background color."""
        r, g, b, *_ = map(float, station_color)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.5 else "white"

    def draw_stations(self, state):
        """Draws all stations at equal angular intervals along the circular route."""
        stations = self._model.type_to_objects["station"]
        num_stations = len(stations)
        delta_theta = 2 * np.pi / num_stations

        for i, station in enumerate(stations):
            theta = i * delta_theta
            x = self.x_center + self.route_radius * np.cos(theta)
            y = self.y_center + self.route_radius * np.sin(theta)

            if station == getattr(self, "depot_station", None):
                station_color = "black"
                text_color = "white"
                self.ax.text(x, y - 0.1, "Depot", fontsize=9, ha="center", color="black", weight="bold")
            else:
                station_color = self.set_station_color(station, state)
                text_color = self.set_text_color(station_color)

            square = Rectangle(
                (x - self.station_size / 2, y - self.station_size / 2),
                self.station_size, self.station_size,
                color=station_color, zorder=3
            )
            self.ax.add_patch(square)
            self.ax.text(x, y, f"{station}", fontsize=8, ha="center", va="center", color=text_color, weight="bold", zorder=4)

    def draw_train(self, train, train_state, station_theta, delta_theta, timer, num_in_queue):
        """
        Draws a train circle and label.
        Position is calculated based on whether the train is:
        - In Route: Interpolated along the arc between stations.
        - In Queue: Offset radially from the station.
        - At Station: Directly on the station square.
        """
        if train_state == "FINISHED": return

        x, y = 0, 0
        
        if train_state in ["WAITING", "ACTIVE"]:
            # On Station
            x = self.x_center + self.route_radius * np.cos(station_theta)
            y = self.y_center + self.route_radius * np.sin(station_theta)
        
        elif train_state == "QUEUE":
            # In Queue: Move inwards radially
            x = self.x_center + (self.route_radius - 0.05 * num_in_queue) * np.cos(station_theta)
            y = self.y_center + (self.route_radius - 0.05 * num_in_queue) * np.sin(station_theta)
        
        elif train_state == "ROUTE":
            # In Route: Arc interpolation
            total_time = self.drive_time_to_dest.get(self.last_station, 30) # Default if not found
            progress = np.clip(1 - (timer / total_time), 0, 1)
            angle = station_theta - (1 - progress) * delta_theta
            
            x = self.x_center + self.route_radius * np.cos(angle)
            y = self.y_center + self.route_radius * np.sin(angle)

        # Draw new markers
        train_circle = Circle((x, y), self.train_size, color="tab:blue", zorder=5)
        train_text = self.ax.text(x + 0.03, y + 0.03, f"{train}", fontsize=9, color="darkblue", weight="bold", zorder=6)
        self.ax.add_patch(train_circle)

    def render(self, state):
        """Main render call: processes state and returns a PIL image."""
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        
        self.create_route_circle()
        self.draw_stations(state)

        trains = self._model.type_to_objects["train"]
        stations = self._model.type_to_objects["station"]
        num_stations = len(stations)
        delta_theta = 2 * np.pi / num_stations

        for train in trains:
            raw_state = state.get(f"current_state___{train}")
            num_q = state.get(f"train_num_at_queue___{train}", 0)
            
            # Map RDDL state to visual categories
            t_state = "ROUTE"
            if raw_state == 4: t_state = "FINISHED"
            elif num_q > 0: t_state = "QUEUE"
            elif raw_state == 2: t_state = "WAITING"
            elif raw_state == 3: t_state = "ACTIVE"

            station = state.get(f"current_station___{train}")
            station_idx = stations.index(station)
            self.last_station = station # Used for interpolation
            
            self.draw_train(
                train, t_state, 
                station_idx * delta_theta, delta_theta,
                state.get(f"train_timer___{train}", 0),
                num_q
            )

        self.fig.canvas.draw()
        return self.convert2img(self.fig.canvas)

    def convert2img(self, canvas):
        """Converts the Matplotlib canvas to a PIL Image."""
        buf, (width, height) = canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        return Image.fromarray(img)
