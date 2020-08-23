
# Check for bash
[ -z "$BASH_VERSION" ] && return

####################################################################################################

__gresource() {
  local choices coffset section

  if [ ${COMP_CWORD} -gt 2 ]; then
      if [ ${COMP_WORDS[1]} = --section ]; then
          section=${COMP_WORDS[2]}
          coffset=2
      else
          coffset=0
      fi
  else
      coffset=0
  fi

  case "$((${COMP_CWORD}-$coffset))" in
    1)
      choices=$'--section \nhelp \nsections \nlist \ndetails \nextract '
      ;;

    2)
      case "${COMP_WORDS[$(($coffset+1))]}" in
        --section)
          return 0
          ;;

        help)
          choices=$'sections\nlist\ndetails\nextract'
          ;;

        sections|list|details|extract)
          COMPREPLY=($(compgen -f -- ${COMP_WORDS[${COMP_CWORD}]}))
          return 0
          ;;
      esac
      ;;

    3)
      case "${COMP_WORDS[$(($coffset+1))]}" in
        list|details|extract)
          choices="$(gresource list ${COMP_WORDS[$(($coffset+2))]} 2> /dev/null | sed -e 's.$. .')"
          ;;
      esac
      ;;
  esac

  local IFS=$'\n'
  COMPREPLY=($(compgen -W "${choices}" -- "${COMP_WORDS[${COMP_CWORD}]}"))
}

####################################################################################################

complete -o nospace -F __gresource gresource
