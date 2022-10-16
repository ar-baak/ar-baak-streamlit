from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import pendulum
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from st_aggrid import (
    AgGrid,
    ColumnsAutoSizeMode,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="阿伯賭馬", layout="wide", menu_items=dict())


class FirestoreRecord(ABC):
    @abstractmethod
    def generate_id(self) -> str:
        pass

    @abstractmethod
    def __eq__(self) -> bool:
        pass


@dataclass
class Meeting(FirestoreRecord):
    meeting_date: datetime = None
    start_time: datetime = None
    num_race: int = 0
    ran_race: int = 0
    venue: str = ""
    ctb_back_url: str = ""
    ctb_lay_url: str = ""
    ended: bool = False
    races: List[firestore.DocumentReference | firestore.AsyncDocumentReference] = field(
        default_factory=lambda: []
    )

    def generate_id(self) -> str:
        return f"meetings/{self.meeting_date:%Y%m%d}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Meeting):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

    @staticmethod
    def from_dict(data_dict: Dict):
        return Meeting(
            meeting_date=data_dict.get("meeting_date", None),
            start_time=data_dict.get("start_time", None),
            num_race=data_dict.get("num_race", None),
            ran_race=data_dict.get("ran_race", None),
            venue=data_dict.get("venue", None),
            ctb_back_url=data_dict.get("ctb_back_url", None),
            ctb_lay_url=data_dict.get("ctb_lay_url", None),
            ended=data_dict.get("ended", None),
            races=data_dict.get("races", None),
        )


@dataclass
class Race(FirestoreRecord):
    num: int = 0
    race_class: str = ""
    course: str = ""
    race_time: datetime = None
    distance: int = None
    race_name: str = ""
    track: str = ""
    venue: str = ""
    field_size: int = 0
    horse_count: int = 0
    reserve_count: int = 0
    ended: bool = False
    end_time: datetime = None
    entries: Dict[
        str, firestore.AsyncDocumentReference | firestore.DocumentReference
    ] = field(default_factory=lambda: dict())
    reserve_entries: List[
        firestore.AsyncDocumentReference | firestore.DocumentReference
    ] = field(default_factory=lambda: [])
    scratched_entries: List[
        firestore.AsyncDocumentReference | firestore.DocumentReference
    ] = field(default_factory=lambda: [])
    meeting: firestore.AsyncDocumentReference | firestore.DocumentReference = None

    def generate_id(self) -> str:
        return f"races/{self.race_time:%Y%m%d}{self.num:02}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Race):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

    @staticmethod
    def race_num_from_id(id: str) -> str:
        return f"Race {id[-2:]}"

    @staticmethod
    def race_num_from_id_ch(id: str) -> str:
        return f"第{id[-2:]}場"

    @staticmethod
    def from_dict(data_dict: Dict):
        return Race(
            num=data_dict.get("num", None),
            race_class=data_dict.get("race_class", None),
            course=data_dict.get("course", None),
            race_time=data_dict.get("race_time", None),
            distance=data_dict.get("distance", None),
            race_name=data_dict.get("race_name", None),
            track=data_dict.get("track", None),
            venue=data_dict.get("venue", None),
            field_size=data_dict.get("field_size", None),
            horse_count=data_dict.get("horse_count", None),
            reserve_count=data_dict.get("reserve_count", None),
            ended=data_dict.get("ended", None),
            end_time=data_dict.get("end_time", None),
            entries=data_dict.get("entries", None),
            scratched_entries=data_dict.get("scratched_entries", None),
            meeting=data_dict.get("meeting", None),
        )


@dataclass
class Entry(FirestoreRecord):
    id: str
    num: int = 0
    draw: int = 0
    horse: firestore.DocumentReference | firestore.AsyncDocumentReference = None
    horse_en: str = ""
    horse_ch: str = ""
    jockey: firestore.DocumentReference | firestore.AsyncDocumentReference = None
    jockey_en: str = ""
    jockey_ch: str = ""
    trainer: firestore.DocumentReference | firestore.AsyncDocumentReference = None
    trainer_en: str = ""
    trainer_ch: str = ""
    horse_weight: int = 0
    handicap_weight: int = 0
    runner_rating: int = 0
    gear: str = ""
    last_six_run: List[int] = field(default_factory=lambda: [])
    saddle_cloth: bool = None
    brand_num: str = ""
    stand_by: bool = None
    apprentice_allowance: str = ""
    scratched: bool = False
    scratched_group: bool | str = False
    members: List = field(default_factory=lambda: [])
    priority: bool = False
    trump_card: bool = False
    preference: int = None
    race: firestore.DocumentReference | firestore.AsyncDocumentReference = None
    latest_hkjc_odds: firestore.DocumentReference | firestore.AsyncDocumentReference = (
        None
    )
    latest_ctb_back_discount: firestore.DocumentReference | firestore.AsyncDocumentReference = (
        None
    )
    latest_ctb_lay_discount: firestore.DocumentReference | firestore.AsyncDocumentReference = (
        None
    )
    latest_odds: firestore.DocumentReference | firestore.AsyncDocumentReference = None

    def generate_id(self) -> str:
        return f"entries/{self.id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Entry):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

    @staticmethod
    def from_dict(data_dict: Dict):
        return Entry(
            id=data_dict.get("id", None),
            num=data_dict.get("num", None),
            draw=data_dict.get("draw", None),
            horse=data_dict.get("horse", None),
            horse_ch=data_dict.get("horse_ch", None),
            horse_en=data_dict.get("horse_en", None),
            jockey=data_dict.get("jockey", None),
            jockey_ch=data_dict.get("jockey_ch", None),
            jockey_en=data_dict.get("jockey_en", None),
            trainer=data_dict.get("trainer", None),
            trainer_ch=data_dict.get("trainer_ch", None),
            trainer_en=data_dict.get("trainer_en", None),
            horse_weight=data_dict.get("horse_weight", None),
            handicap_weight=data_dict.get("handicap_weight", None),
            runner_rating=data_dict.get("runner_rating", None),
            gear=data_dict.get("gear", None),
            last_six_run=data_dict.get("last_six_run", None),
            saddle_cloth=data_dict.get("saddle_cloth", None),
            brand_num=data_dict.get("brand_num", None),
            stand_by=data_dict.get("stand_by", None),
            apprentice_allowance=data_dict.get("apprentice_allowance", None),
            scratched=data_dict.get("scratched", None),
            scratched_group=data_dict.get("scratched_group", None),
            members=data_dict.get("members", None),
            priority=data_dict.get("priority", None),
            trump_card=data_dict.get("trump_card", None),
            preference=data_dict.get("preference", None),
            race=data_dict.get("race", None),
            latest_hkjc_odds=data_dict.get("latest_hkjc_odds", None),
            latest_ctb_back_discount=data_dict.get("latest_ctb_back_discount", None),
            latest_ctb_lay_discount=data_dict.get("latest_ctb_lay_discount", None),
            latest_odds=data_dict.get("latest_odds", None),
        )


@dataclass
class Horse(FirestoreRecord):
    id: str
    name_en: str = ""
    name_ch: str = ""
    entries: List[
        firestore.DocumentReference | firestore.AsyncDocumentReference
    ] = field(default_factory=lambda: [])

    def generate_id(self) -> str:
        return f"horses/{self.id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Horse):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

    @staticmethod
    def from_dict(data_dict: Dict):
        return Horse(
            id=data_dict.get("id", None),
            name_en=data_dict.get("name_en", None),
            name_ch=data_dict.get("name_ch", None),
            entries=data_dict.get("entries", []),
        )


@dataclass
class Jockey(FirestoreRecord):
    id: str
    name_en: str = ""
    name_ch: str = ""
    entries: List[
        firestore.DocumentReference | firestore.AsyncDocumentReference
    ] = field(default_factory=lambda: [])

    def generate_id(self) -> str:
        return f"jockeys/{self.id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Jockey):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

    @staticmethod
    def from_dict(data_dict: Dict):
        return Jockey(
            id=data_dict.get("id", None),
            name_en=data_dict.get("name_en", None),
            name_ch=data_dict.get("name_ch", None),
            entries=data_dict.get("entries", []),
        )


def safe_cast_int_from_str(s: str, default: int = 0) -> int:
    return int(s) if s != "" else default


@dataclass
class Trainer(FirestoreRecord):
    id: str
    name_en: str = ""
    name_ch: str = ""
    entries: List[
        firestore.DocumentReference | firestore.AsyncDocumentReference
    ] = field(default_factory=lambda: [])

    def generate_id(self) -> str:
        return f"trainers/{self.id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trainer):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

    @staticmethod
    def from_dict(data_dict: Dict):
        return Trainer(
            id=data_dict.get("id", None),
            name_en=data_dict.get("name_en", None),
            name_ch=data_dict.get("name_ch", None),
            entries=data_dict.get("entries", []),
        )


@st.cache(ttl=5)
def get_entries_of_race(race_path):
    st.session_state.race_path = race_path
    st.session_state.get_entries_of_race = True
    race_ref = db.document(st.session_state.race_path)
    race = Race.from_dict(race_ref.get().to_dict())
    st.session_state.race = race
    entry_refs = race.entries
    # print(entry_refs.to_dict())
    entries = []
    for _, entry_ref in entry_refs.items():
        data = entry_ref.get().to_dict()
        if data["latest_ctb_back_discount"] is not None:
            ctb_back = data["latest_ctb_back_discount"].get().to_dict()
        else:
            ctb_back = dict()
        if data["latest_ctb_lay_discount"] is not None:
            ctb_lay = data["latest_ctb_lay_discount"].get().to_dict()
        else:
            ctb_lay = dict()
        if data["latest_hkjc_odds"] is not None:
            hkjc_odds = data["latest_hkjc_odds"].get().to_dict()
        else:
            hkjc_odds = dict()
        data["ctb_back_win_discount"] = ctb_back.get("win", None)
        data["ctb_back_win_amount"] = ctb_back.get("win_amount", None)
        data["ctb_back_place_discount"] = ctb_back.get("place", None)
        data["ctb_back_place_amount"] = ctb_back.get("place_amount", None)
        data["ctb_lay_win_discount"] = ctb_lay.get("win", None)
        data["ctb_lay_win_amount"] = ctb_lay.get("win_amount", None)
        data["ctb_lay_place_discount"] = ctb_lay.get("place", None)
        data["ctb_lay_place_amount"] = ctb_lay.get("place_amount", None)
        data["hkjc_odds_win"] = hkjc_odds.get("win", None)
        data["hkjc_odds_place"] = hkjc_odds.get("place", None)
        entries.append(
            {
                "No.": data["num"],
                "Draw": data["draw"],
                "Name": data["horse_ch"],
                "Jockey": data["jockey_ch"],
                "Trainer": data["trainer_ch"],
                "Win Odds": data["hkjc_odds_win"],
                "Place Odds": data["hkjc_odds_place"],
                "Lay Win Discount": data["ctb_lay_win_discount"],
                "Lay Place Discount": data["ctb_lay_place_discount"],
                # "Back Win Discount": data["ctb_back_win_discount"],
                # "Back Place Discount": data["ctb_back_place_discount"],
            }
        )
    st.session_state.entries = pd.DataFrame(entries).sort_values(by="No.")
    st.session_state.get_entries_of_race = False


if "entries" not in st.session_state:
    st.session_state.entries = pd.DataFrame()
if "race_path" not in st.session_state:
    st.session_state.race_path = None
if "race" not in st.session_state:
    st.session_state.race = None
if "get_entries_of_race" not in st.session_state:
    st.session_state.get_entries_of_race = False

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
# Authenticate to Firestore with the JSON account key.
db = firestore.Client(credentials=credentials)

today = pendulum.today(tz="Asia/Hong_Kong")
# Create a reference to the Google post.
doc_ref = db.collection("meetings").document(f"{today:%Y%m%d}")

# Then get the data at that reference.
doc = doc_ref.get()

if not doc.exists:
    st.write("今日冇馬跑呀 訓啦柒頭")
else:
    # Let's see what we got!
    st.write("今日係" + today.format("Y年 M月 D日 dddd", locale="zh"))
    meeting = Meeting.from_dict(doc.to_dict())

    autorefresh = st_autorefresh(interval=10000)

    if autorefresh:
        if st.session_state.race_path and not st.session_state.get_entries_of_race:
            get_entries_of_race(st.session_state.race_path)

    for col, race in zip(st.columns(len(meeting.races)), meeting.races):
        col.button(
            Race.race_num_from_id_ch(race.path),
            on_click=get_entries_of_race,
            args=(race.path,),
        )

    if "entries" in st.session_state:
        if len(st.session_state.entries) > 0:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(
                label="第幾場",
                value=Race.race_num_from_id_ch(st.session_state.race.generate_id()),
            )
            col2.metric(
                label="幾點跑",
                value=pendulum.from_timestamp(
                    st.session_state.race.race_time.timestamp()
                )
                .in_timezone("Asia/Hong_Kong")
                .format("HH點mm分"),
            )
            col3.metric(
                label="咩班",
                value=st.session_state.race.race_class,
            )
            col4.metric(
                label="咩地",
                value=st.session_state.race.track,
            )
            col5.metric(
                label="跑幾遠",
                value=f"{st.session_state.race.distance}米",
            )
            AgGrid(
                st.session_state.entries,
                width="100%",
                fit_columns_on_grid_load=True,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                theme="material",
            )
