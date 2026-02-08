import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from pyRDDLGym.core.compiler.model import RDDLPlanningModel


class TrainRouteVisualizer:
    def __init__(self, model):
        self._model = model

        # Reduced figsize and set DPI to avoid exceeding maximum image size
        self.fig, self.ax = plt.subplots(
            figsize=(5, 5), dpi=80
        )  # Adjusted figsize and DPI
        self.route_radius = 0.4  # Set the route radius
        self.train_radius = self.route_radius - 0.1  # Slightly inside the route
        self.station_size = 0.05  # Adjust station square size as needed
        self.train_size = 0.01  # Adjust train size
        self.x_center = 0.5  # Center of the route
        self.y_center = 0.5  # Center of the route
        self._nonfluents = model.ground_vars_with_values(
            model.non_fluents
        )  # Load all non-fluents

        self.train_patches = {}  # Store train circles by train ID

        # Create the route circle
        self.create_route_circle()

        # store drive time to stations
        self.drive_time_to_dest = {}
        for k, v in self._nonfluents.items():
            var, objects = RDDLPlanningModel.parse_grounded(k)
            if var == "DRIVE_TIME":
                _, dest = objects  # Only care about the destination
                if v > 0:
                    self.drive_time_to_dest[dest] = v  # Store time to destination

        # store depot station
        for k, v in self._nonfluents.items():
            var, objects = RDDLPlanningModel.parse_grounded(k)
            if var == "DEPOT_STATION":
                if v:
                    self.depot_station = objects[0]  # Store the depot station

    def get_drive_time_to_dest(self, dest):
        return self.drive_time_to_dest.get(dest, -1)  # Default to -1 if not found

    def create_route_circle(self):
        """Draw the circular route."""
        route_circle = plt.Circle(
            (self.x_center, self.y_center),
            self.route_radius,
            color="blue",
            fill=False,
            linestyle="dashed",
        )
        self.ax.add_patch(route_circle)

    # set the station color such that it shows the crowding level
    def set_station_color(self, station, state):
        capacity = 1000
        passengers_at_station = state.get(f"passengers_at_station___{station}", 0)
        crowding_level = passengers_at_station / capacity

        cmap = plt.get_cmap("RdYlGn_r")
        color = cmap(crowding_level)
        return color

    # set the "s1/s2/s3.." text, so it will be seen regardless to station color
    def set_text_color(self, station_color):
        r, g, b, *_ = map(float, station_color)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b  # Calculate luminance

        # Return black for bright colors, white for dark colors
        return "black" if luminance > 0.5 else "white"

    def draw_stations(self, state):
        """Draw stations equally spaced along the route."""
        stations = self._model.type_to_objects["station"]
        num_stations = len(stations)
        delta_theta = 2 * np.pi / num_stations  # Angle between stations

        for i, station in enumerate(stations):

            theta = i * delta_theta  # Compute station angle
            x = self.x_center + self.route_radius * np.cos(theta)
            y = self.y_center + self.route_radius * np.sin(theta)

            if station == self.depot_station:
                station_color = "black"
                text_color = "white"
                self.ax.text(
                    x,
                    y - 0.1,
                    "Depot",
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    color="red",
                )

            else:
                station_color = self.set_station_color(station, state)
                text_color = self.set_text_color(station_color)

            # Create and draw station (square)
            square = Rectangle(
                (x - self.station_size / 2, y - self.station_size / 2),
                self.station_size,
                self.station_size,
                color=station_color,
            )
            self.ax.add_patch(square)

            # Add station number inside the station
            self.ax.text(
                x,
                y - self.station_size / 2,
                f"{station}",
                fontsize=10,
                ha="center",
                va="bottom",
                color=text_color,
            )

    def draw_train(
        self,
        last_train_state,
        train_state,
        train,
        station,
        station_theta,
        delta_theta,
        train_timer,
        num_in_queue,
    ):
        """Draw the train on the route based on its state."""

        # delete last draw of train
        if self.train_patches.get(train) is not None:
            old_patch, old_text = self.train_patches[train]
            old_patch.remove()  # Removes the train's old circle from the figure
            old_text.remove()  # Remove previous train label
            del self.train_patches[train]

        # delete trains at the end
        if train_state == "train_finished":
            return

        # if train in now in station, draw it in station
        if train_state in ["train_waiting_at_station", "train_active_at_station"]:
            x = self.x_center + self.route_radius * np.cos(station_theta)
            y = self.y_center + self.route_radius * np.sin(station_theta)
            train_circle = Circle((x, y), self.train_size)
            x_text = x + 0.07 * np.cos(station_theta)
            y_text = y + 0.07 * np.sin(station_theta)
            train_text = self.ax.text(
                x_text,
                y_text,
                f"{train}",
                fontsize=10,
                ha="left",
                va="center",
                color="black",
            )
            self.ax.add_patch(train_circle)
            self.train_patches[train] = (train_circle, train_text)

        if train_state == "train_num_at_queue":

            x = self.x_center + (self.route_radius - 0.05 * num_in_queue) * np.cos(
                station_theta
            )
            y = self.y_center + (self.route_radius - 0.05 * num_in_queue) * np.sin(
                station_theta
            )

            train_circle = Circle((x, y), self.train_size)
            x_text = x + 0.065 * np.cos(station_theta + np.pi / 2)
            y_text = y + 0.065 * np.sin(station_theta + np.pi / 2)
            train_text = self.ax.text(
                x_text,
                y_text,
                f"{train}",
                fontsize=10,
                ha="left",
                va="center",
                color="black",
            )
            self.ax.add_patch(train_circle)
            self.train_patches[train] = (train_circle, train_text)

        if train_state == "train_in_route":

            # if the train is in route to the first station, and the time is longer than the

            total_drive_time = self.get_drive_time_to_dest(station)
            progress_theta = delta_theta / total_drive_time

            if train_timer <= total_drive_time:

                x = self.x_center + self.route_radius * np.cos(
                    station_theta - progress_theta * (train_timer)
                )
                y = self.y_center + self.route_radius * np.sin(
                    station_theta - progress_theta * (train_timer)
                )

                train_circle = Circle((x, y), self.train_size)
                train_text = self.ax.text(
                    x + 0.05,
                    y,
                    f"{train}",
                    fontsize=10,
                    ha="left",
                    va="center",
                    color="black",
                )
                self.ax.add_patch(train_circle)
                self.train_patches[train] = (train_circle, train_text)

    def render(self, state):
        """Render the train route visualization."""
        trains = self._model.type_to_objects["train"]
        stations = self._model.type_to_objects["station"]
        num_stations = len(stations)
        delta_theta = 2 * np.pi / num_stations  # Angle between stations

        self.draw_stations(state)

        last_train_state = {train: None for train in trains}
        # Iterate through trains and stations
        for train in trains:

            # Determine the state of the train for this station
            train_state = None
            if state.get(f"current_state___{train}") == 0:
                train_state = "train_in_route"
            elif state.get(f"train_num_at_queue___{train}") > 0:
                train_state = "train_num_at_queue"
            elif state.get(f"current_state___{train}") == 2:
                train_state = "train_waiting_at_station"
            elif state.get(f"current_state___{train}") == 3:
                train_state = "train_active_at_station"
            elif state.get(f"current_state___{train}") == 4:
                train_state = "train_finished"

            station = state.get(f"current_station___{train}")

            if train_state:

                # Compute station angle on the circle
                station_index = stations.index(station)
                station_theta = station_index * delta_theta

                # train timer
                train_timer = state.get(f"train_timer___{train}", 0)

                # train num at queue
                num_in_queue = state.get(f"train_num_at_queue___{train}", 0)

                # Draw train
                self.draw_train(
                    last_train_state,
                    train_state,
                    train,
                    station,
                    station_theta,
                    delta_theta,
                    train_timer,
                    num_in_queue,
                )
                last_train_state[train] = train_state

        # Finalize and return image
        self.fig.canvas.draw()
        img = self.convert2img(self.fig.canvas)
        return img

    def convert2img(self, canvas):
        """Convert Matplotlib figure to a PIL Image."""
        canvas.draw()
        buf, (width, height) = canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(
            (height, width, 4)
        )  # RGBA format
        return Image.fromarray(img)
