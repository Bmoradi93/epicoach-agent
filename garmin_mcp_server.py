#!/usr/bin/env python3
"""
Garmin Connect MCP Server (FastMCP)

A modern MCP (Model Context Protocol) server using FastMCP that provides access to 
Garmin Connect data through standardized tools. This server exposes the full 
functionality of the python-garminconnect library as MCP tools.

Usage:
    python garmin_mcp_server.py

Environment Variables:
    GARMIN_EMAIL: Your Garmin Connect email
    GARMIN_PASSWORD: Your Garmin Connect password
    GARMINTOKENS: Path to store authentication tokens (default: ~/.garminconnect)
"""

import os
import logging
from datetime import date, datetime
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import Garmin Connect library
try:
    from garminconnect import (
        Garmin,
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
        GarminConnectInvalidFileFormatError,
    )
except ImportError:
    print("Error: garminconnect library not found. Please install it first:")
    print("pip install garminconnect")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("garmin_mcp_server")

# Initialize FastMCP server
mcp = FastMCP("Garmin Connect MCP Server")

# Get credentials from environment
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")
GARMIN_TOKENSTORE = os.getenv("GARMINTOKENS", "~/.garminconnect")

# Global Garmin API instance
garmin_api: Optional[Garmin] = None
is_authenticated = False

# Auto-authenticate on startup
def _auto_authenticate():
    """Automatically authenticate on server startup"""
    global garmin_api, is_authenticated
    try:
        if GARMIN_EMAIL and GARMIN_PASSWORD:
            garmin_api = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
            garmin_api.login()
            is_authenticated = True
            logger.info("Successfully auto-authenticated with Garmin Connect")
        else:
            logger.warning("No Garmin credentials found in environment variables")
    except Exception as e:
        logger.error(f"Auto-authentication failed: {e}")
        is_authenticated = False

# Skip auto-authentication - let garmin_login tool handle it
# _auto_authenticate()


class LoginRequest(BaseModel):
    """Request model for Garmin login"""
    email: Optional[str] = Field(None, description="Garmin Connect email (optional if set in env)")
    password: Optional[str] = Field(None, description="Garmin Connect password (optional if set in env)")
    tokenstore: Optional[str] = Field(None, description="Token storage path (optional)")


class DateRequest(BaseModel):
    """Request model for date-based queries"""
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format (default: today)")


class DateRangeRequest(BaseModel):
    """Request model for date range queries"""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format (optional)")


class ActivitiesRequest(BaseModel):
    """Request model for activities queries"""
    start: Optional[int] = Field(0, description="Starting offset (default: 0)")
    limit: Optional[int] = Field(20, description="Number of activities to return (default: 20)")
    activity_type: Optional[str] = Field(None, description="Filter by activity type (cycling, running, swimming, etc.)")


class ActivityDetailsRequest(BaseModel):
    """Request model for activity details"""
    activity_id: str = Field(..., description="Activity ID")


class ActivitiesByDateRequest(BaseModel):
    """Request model for activities by date range"""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format (optional)")
    activity_type: Optional[str] = Field(None, description="Filter by activity type (optional)")


class WeighInRequest(BaseModel):
    """Request model for adding weight measurements"""
    weight: float = Field(..., description="Weight value")
    unit: Optional[str] = Field("kg", description="Weight unit (default: kg)")
    timestamp: Optional[str] = Field(None, description="Timestamp in ISO format (optional, defaults to now)")


@mcp.tool()
def test_login(random_string: str = "dummy") -> Dict[str, Any]:
    """Test authentication function"""
    return {"status": "test", "message": "Test function works"}

@mcp.tool()
def garmin_login(random_string: str = "dummy") -> Dict[str, Any]:
    """Authenticate with Garmin Connect using environment variables"""
    global garmin_api, is_authenticated
    
    try:
        # Use garth directly for authentication
        from garth import Client as GarthClient
        garth_client = GarthClient()
        garth_client.login(GARMIN_EMAIL, GARMIN_PASSWORD)
        
        # Now create Garmin instance with authenticated garth client
        garmin_api = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
        garmin_api.garth = garth_client  # Use the authenticated garth client
        is_authenticated = True
        
        return {
            "status": "success",
            "message": "Successfully authenticated with Garmin Connect"
        }
        
    except Exception as e:
        is_authenticated = False
        return {
            "status": "error", 
            "message": f"Authentication failed: {str(e)}"
        }


def _check_authentication():
    """Check if user is authenticated, raise exception if not"""
    if not is_authenticated or not garmin_api:
        raise Exception("Not authenticated. Please call garmin_login first.")


@mcp.tool()
def get_user_summary(date: str = "") -> Dict[str, Any]:
    """Get user activity summary for a specific date (YYYY-MM-DD format, default: today)"""
    _check_authentication()
    date_str = date if date else date.today().isoformat()
    return garmin_api.get_user_summary(date_str)


@mcp.tool()
def get_user_profile(random_string: str = "dummy") -> Dict[str, Any]:
    """Get user profile and settings"""
    _check_authentication()
    return garmin_api.get_user_profile()


@mcp.tool()
def get_steps_data(date_param: str = "") -> Dict[str, Any]:
    """Get steps data for a specific date (YYYY-MM-DD format, default: today)"""
    _check_authentication()
    date_str = date_param if date_param else date.today().isoformat()
    return garmin_api.get_steps_data(date_str)


@mcp.tool()
def get_heart_rate_data(date_param: str = "") -> Dict[str, Any]:
    """Get heart rate data for a specific date (YYYY-MM-DD format, default: today)"""
    _check_authentication()
    date_str = date_param if date_param else date.today().isoformat()
    return garmin_api.get_heart_rates(date_str)


@mcp.tool()
def get_sleep_data(date_param: str = "") -> Dict[str, Any]:
    """Get sleep data for a specific date (YYYY-MM-DD format, default: today)"""
    _check_authentication()
    date_str = date_param if date_param else date.today().isoformat()
    return garmin_api.get_sleep_data(date_str)


@mcp.tool()
def get_stress_data(date_param: str = "") -> Dict[str, Any]:
    """Get stress data for a specific date (YYYY-MM-DD format, default: today)"""
    _check_authentication()
    date_str = date_param if date_param else date.today().isoformat()
    return garmin_api.get_stress_data(date_str)


@mcp.tool()
def get_body_battery(start_date: str, end_date: str = "") -> Dict[str, Any]:
    """Get body battery data for a date range (YYYY-MM-DD format)"""
    _check_authentication()
    if end_date:
        return garmin_api.get_body_battery(start_date, end_date)
    else:
        return garmin_api.get_body_battery(start_date)


@mcp.tool()
def get_activities(start: int = 0, limit: int = 20, activity_type: str = "") -> List[Dict[str, Any]]:
    """Get list of activities with optional filtering"""
    _check_authentication()
    activity_filter = activity_type if activity_type else None
    return garmin_api.get_activities(start, limit, activity_filter)


@mcp.tool()
def get_activity_details(activity_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific activity"""
    _check_authentication()
    return garmin_api.get_activity_details(activity_id)


@mcp.tool()
def get_activities_by_date(start_date: str, end_date: str = "", activity_type: str = "") -> List[Dict[str, Any]]:
    """Get activities for a specific date range (YYYY-MM-DD format)"""
    _check_authentication()
    end_date_param = end_date if end_date else None
    activity_type_param = activity_type if activity_type else None
    return garmin_api.get_activities_by_date(start_date, end_date_param, activity_type_param)


@mcp.tool()
def get_devices() -> List[Dict[str, Any]]:
    """Get list of registered Garmin devices"""
    _check_authentication()
    return garmin_api.get_devices()


@mcp.tool()
def add_weigh_in(weight: float, unit: str = "kg", timestamp: str = "") -> Dict[str, Any]:
    """Add a weight measurement"""
    _check_authentication()
    
    try:
        if timestamp and timestamp.strip():
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            ts = datetime.now()
        
        # Try passing timestamp as ISO string if datetime object fails
        result = garmin_api.add_weigh_in(weight, unit, ts.isoformat())
        
        # Handle response object properly
        if hasattr(result, 'status_code'):
            response_data = {
                "status_code": result.status_code,
                "success": result.status_code in [200, 201, 204]  # 204 = No Content (success for add operations)
            }
        else:
            response_data = str(result)
            
        return {
            "status": "success",
            "message": "Weight measurement added successfully",
            "result": response_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to add weight measurement: {str(e)}"
        }


@mcp.tool()
def download_activity(activity_id: str, format: str = "GPX") -> Dict[str, Any]:
    """Download an activity in a requested format (GPX, TCX, CSV, KML, ORIGINAL) and save it locally.
    
    Args:
        activity_id (str): The unique activity ID.
        format (str, optional): The download format. Supported values: "GPX", "TCX", "CSV", "KML", "ORIGINAL". Defaults to "GPX".
        
    Returns:
        dict: Status, local file path, and file size.
    """
    _check_authentication()
    
    try:
        fmt_upper = format.upper()
        if fmt_upper == "GPX":
            dl_fmt = Garmin.ActivityDownloadFormat.GPX
            ext = "gpx"
        elif fmt_upper == "TCX":
            dl_fmt = Garmin.ActivityDownloadFormat.TCX
            ext = "tcx"
        elif fmt_upper == "CSV":
            dl_fmt = Garmin.ActivityDownloadFormat.CSV
            ext = "csv"
        elif fmt_upper == "KML":
            dl_fmt = Garmin.ActivityDownloadFormat.KML
            ext = "kml"
        elif fmt_upper == "ORIGINAL":
            dl_fmt = Garmin.ActivityDownloadFormat.ORIGINAL
            ext = "zip"
        else:
            return {
                "status": "error",
                "message": f"Unsupported format: {format}. Supported formats are GPX, TCX, CSV, KML, ORIGINAL."
            }
            
        data = garmin_api.download_activity(activity_id, dl_fmt=dl_fmt)
        
        # Save file locally
        filename = f"activity_{activity_id}.{ext}"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, "wb") as f:
            f.write(data)
            
        return {
            "status": "success",
            "message": f"Activity downloaded successfully",
            "file_path": filepath,
            "file_name": filename,
            "file_size_bytes": len(data)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to download activity: {str(e)}"
        }


def test_direct():
    """Test function for debugging"""
    from garminconnect import Garmin
    import os
    
    email = os.getenv('GARMIN_EMAIL')
    password = os.getenv('GARMIN_PASSWORD')
    print(f'Testing with email: {email}')
    
    api = Garmin(email, password)
    api.login()
    print('✅ Direct authentication successful!')
    
    activities = api.get_activities(0, 1)
    if activities:
        last = activities[0]
        print(f'Last activity: {last.get("activityName", "Unknown")} on {last.get("startTimeLocal", "Unknown")}')
    return activities

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_direct()
    else:
        # Run the MCP server
        mcp.run()