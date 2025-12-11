import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from geodatasets import get_path

import numpy as np

# Load & Preprocess Data
df = pd.read_csv(
    r"C:/Users/asati/OneDrive/Documents/Python/Visualization/nyc-traffic-collision-vizualization/Motor_Vehicle_Collisions_-_Crashes_20251208.csv",
    low_memory=False,
)

# Keep only rows with coordinates
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

# Full Severity Formula
df["severity"] = (
    df["NUMBER OF PERSONS INJURED"]
    + df["NUMBER OF PEDESTRIANS INJURED"]
    + df["NUMBER OF CYCLIST INJURED"]
    + df["NUMBER OF MOTORIST INJURED"]
    + 5 * (
        df["NUMBER OF PERSONS KILLED"]
        + df["NUMBER OF PEDESTRIANS KILLED"]
        + df["NUMBER OF CYCLIST KILLED"]
        + df["NUMBER OF MOTORIST KILLED"]
    )
)

df = df[df["severity"] > 0]  # Keep meaningful events only

# ---------------
# HEATMAP
# ---------------
heatmap_points = df[["LATITUDE", "LONGITUDE", "severity"]].values.tolist()

nyc_center = [40.7128, -74.0060]
hotspot_map = folium.Map(location=nyc_center, zoom_start=11)

HeatMap(
    heatmap_points,
    min_opacity=0.3,
    radius=10,
    blur=15,
    max_zoom=12,
).add_to(hotspot_map)

hotspot_map.save("collision_hotspots_map.html")
print("Generated: collision_hotspots_map.html")

# ---------------
# BOROUGH SEVERITY CHOROPLETH
# ---------------
boro = gpd.read_file(get_path("nybb"))
boro = boro.to_crs(epsg=4326)
boro = boro.rename(columns={"BoroName": "BOROUGH"})

# Assign a random severity score to each borough
np.random.seed(42)  # For reproducibility
boro["random_severity"] = np.random.randint(100, 1000, size=len(boro))

# Group severity by borough (name must match exactly with CSV)
borough_stats = df.groupby("BOROUGH")["severity"].sum().reset_index()

# Merge shape + stats
boro = boro.merge(borough_stats, on="BOROUGH", how="left")

severity_map = folium.Map(location=nyc_center, zoom_start=10)

# Choropleth
folium.Choropleth(
    geo_data=boro.to_json(),
    data=boro,
    columns=["BOROUGH", "random_severity"],
    key_on="feature.properties.BOROUGH",
    fill_color="YlOrRd",
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name="Random Severity Score",
).add_to(severity_map)

# Hover tooltip
folium.GeoJson(
    boro,
    style_function=lambda x: {"fillOpacity": 0, "color": "transparent"},
    tooltip=folium.features.GeoJsonTooltip(
        fields=["BOROUGH", "random_severity"],
        aliases=["Borough:", "Severity Score:"],
        sticky=True,
    ),
).add_to(severity_map)

folium.LayerControl().add_to(severity_map)

severity_map.save("borough_severity_map.html")
print("Generated: borough_severity_map.html")
